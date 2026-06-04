import re
import time
from typing import List

from fastapi import FastAPI, Request
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import Meter, NoOpMeter
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import AggregationTemporality, PeriodicExportingMetricReader

from letta.helpers.datetime_helpers import ns_to_ms
from letta.log import get_logger
from letta.otel.context import add_ctx_attribute, get_ctx_attributes
from letta.otel.resource import get_resource, is_pytest_environment
from letta.settings import settings

logger = get_logger(__name__)

_meter: Meter = NoOpMeter("noop")
_is_metrics_initialized: bool = False

# Endpoints to include in endpoint metrics tracking (opt-in) vs tracing.py opt-out
_included_v1_endpoints_regex: List[str] = [
    "^POST /v1/agents/(?P<agent_id>[^/]+)/messages$",
    "^POST /v1/agents/(?P<agent_id>[^/]+)/messages/stream$",
    "^POST /v1/agents/(?P<agent_id>[^/]+)/messages/async$",
]

# Header attributes to set context with
header_attributes = {
    "x-organization-id": "organization.id",
    "x-project-id": "project.id",
    "x-base-template-id": "base_template.id",
    "x-template-id": "template.id",
    "x-agent-id": "agent.id",
}


async def _otel_metric_middleware(request: Request, call_next):
    if not _is_metrics_initialized:
        return await call_next(request)

    for header_key, otel_key in header_attributes.items():
        header_value = request.headers.get(header_key)
        if header_value:
            add_ctx_attribute(otel_key, header_value)

    # Opt-in check for latency / error tracking
    endpoint_path = f"{request.method} {request.url.path}"
    should_track_endpoint_metrics = any(re.match(regex, endpoint_path) for regex in _included_v1_endpoints_regex)

    if not should_track_endpoint_metrics:
        return await call_next(request)

    # --- Opt-in endpoint metrics ---
    start_perf_counter_ns = time.perf_counter_ns()
    response = None
    status_code = 500  # reasonable default

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as e:
        # Determine status code from exception
        status_code = getattr(e, "status_code", 500)
        raise
    finally:
        end_to_end_ms = ns_to_ms(time.perf_counter_ns() - start_perf_counter_ns)
        _record_endpoint_metrics(
            request=request,
            latency_ms=end_to_end_ms,
            status_code=status_code,
        )


def _record_endpoint_metrics(
    request: Request,
    latency_ms: float,
    status_code: int,
):
    """Record endpoint latency and request count metrics."""
    try:
        # Get the route pattern for better endpoint naming
        route = request.scope.get("route")
        endpoint_name = route.path if route and hasattr(route, "path") else "unknown"

        attrs = {
            "endpoint_path": endpoint_name,
            "method": request.method,
            "status_code": status_code,
            **get_ctx_attributes(),
        }
        from letta.otel.metric_registry import MetricRegistry

        MetricRegistry().endpoint_e2e_ms_histogram.record(latency_ms, attributes=attrs)
        MetricRegistry().endpoint_request_counter.add(1, attributes=attrs)

    except Exception as e:
        logger.warning(f"Failed to record endpoint metrics: {e}")


def setup_metrics(
    endpoint: str | None = None,
    app: FastAPI | None = None,
    service_name: str = "memgpt-server",
    prometheus_enabled: bool = False,
) -> None:
    """Initialise the metrics pipeline.

    Two independent readers may be attached to the MeterProvider:
      - OTLP push exporter (when `endpoint` is set) — ships metrics to a collector.
      - Prometheus pull reader (when `prometheus_enabled`) — serves /metrics from a
        standalone HTTP server on its OWN dedicated port (settings.otel_metrics_prometheus_port),
        NOT the API port, so infra can firewall it independently of public traffic.
        Cumulative temporality is forced by the reader, as Prometheus requires;
        this is independent of otel_preferred_temporality.

    At least one of `endpoint` / `prometheus_enabled` must be provided, otherwise
    metrics stay uninitialised (and all MetricRegistry instruments are no-ops).
    """
    if is_pytest_environment():
        return

    from letta.otel import prom_multiprocess

    prom_port = settings.otel_metrics_prometheus_port
    prom_addr = settings.otel_metrics_prometheus_addr

    # Multi-worker aggregation path: when PROMETHEUS_MULTIPROC_DIR is set, the scrape
    # endpoint is served by native prometheus_client instruments via a MultiProcessCollector
    # (see prom_multiprocess.py), which aggregates across ALL uvicorn workers. This runs
    # alongside OTel; the OTel instruments/OTLP push below are unaffected.
    use_multiprocess = prometheus_enabled and prom_multiprocess.multiprocess_enabled()

    # In-process OTel pull endpoint serves only THIS process. It's valid at a single worker;
    # with >1 worker (and no multiprocess dir) only one worker could bind the port and would
    # expose partial, misleading metrics, so we refuse to start it and explain the options.
    otel_pull_active = prometheus_enabled and not use_multiprocess
    if otel_pull_active and settings.uvicorn_workers > 1:
        logger.warning(
            f"Prometheus pull endpoint DISABLED: it cannot aggregate across {settings.uvicorn_workers} "
            f"uvicorn workers (only one could bind port {prom_port}, exposing partial/misleading metrics). "
            f"For complete multi-worker metrics set PROMETHEUS_MULTIPROC_DIR (native aggregation across "
            f"workers), or run LETTA_UVICORN_WORKERS=1, or push via LETTA_OTEL_EXPORTER_OTLP_ENDPOINT."
        )
        otel_pull_active = False

    metric_readers = []

    if endpoint:
        preferred_temporality = AggregationTemporality(settings.otel_preferred_temporality)
        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=endpoint,
            preferred_temporality={
                # Add more as needed here.
                Counter: preferred_temporality,
                Histogram: preferred_temporality,
            },
        )
        metric_readers.append(PeriodicExportingMetricReader(exporter=otlp_metric_exporter))

    if otel_pull_active:
        # Registers a collector into the default prometheus_client REGISTRY; the
        # standalone server started below serves that same registry.
        from opentelemetry.exporter.prometheus import PrometheusMetricReader

        metric_readers.append(PrometheusMetricReader())

    global _is_metrics_initialized, _meter

    # Stand up the OTel MeterProvider if any OTel reader is configured. (In multiprocess mode
    # with no OTLP endpoint there may be none — that's fine, the native scrape path below still
    # runs; OTel instruments simply stay no-ops for this process.)
    if metric_readers:
        meter_provider = MeterProvider(resource=get_resource(service_name), metric_readers=metric_readers)
        metrics.set_meter_provider(meter_provider)
        _meter = metrics.get_meter(__name__)
        if app:
            app.middleware("http")(_otel_metric_middleware)
        _is_metrics_initialized = True
    elif not use_multiprocess:
        logger.warning("setup_metrics called with no exporter configured (no OTLP endpoint, Prometheus disabled); skipping.")
        return

    # Start the dedicated-port scrape server. Unauthenticated by design — restrict
    # reachability at the network layer.
    if use_multiprocess:
        if prom_multiprocess.start_multiprocess_metrics_server(prom_port, prom_addr):
            logger.info(
                f"Prometheus multiprocess metrics server listening on {prom_addr}:{prom_port} "
                f"(path /metrics; aggregates all {settings.uvicorn_workers} workers)"
            )
    elif otel_pull_active:
        from prometheus_client import start_http_server

        try:
            start_http_server(port=prom_port, addr=prom_addr)
            logger.info(f"Prometheus metrics scrape server listening on {prom_addr}:{prom_port} (path /metrics)")
        except OSError as e:
            logger.warning(f"Could not bind Prometheus metrics server on {prom_addr}:{prom_port}: {e}.")


def get_letta_meter() -> Meter:
    """Returns the global letta meter if metrics are initialized."""
    if not _is_metrics_initialized or isinstance(_meter, NoOpMeter):
        logger.warning("Metrics are not initialized or meter is not available.")
    return _meter
