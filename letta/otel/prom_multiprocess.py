"""Native prometheus_client metrics that aggregate across uvicorn workers.

This runs ALONGSIDE the OpenTelemetry pipeline and does not touch it. OTel instruments,
the OTLP push exporter, and tracing are all untouched and keep working exactly as before.

Why this exists: prometheus_client's multiprocess mode (PROMETHEUS_MULTIPROC_DIR +
MultiProcessCollector) only aggregates *native* prometheus_client instruments — it cannot
see OpenTelemetry instruments exposed via PrometheusMetricReader (those live in each
worker's in-process OTel SDK). So to get a single scrape endpoint that reflects ALL uvicorn
workers, we mirror the headline latency metric onto native prometheus_client instruments
here. Each worker writes to the shared PROMETHEUS_MULTIPROC_DIR; one worker serves a
MultiProcessCollector registry that reads every worker's files and returns the aggregate.

Activation: only when the PROMETHEUS_MULTIPROC_DIR environment variable is set (the standard
prometheus_client multiprocess switch). When unset, every function here is an inert no-op, so
single-worker / OTLP-only deployments are completely unaffected.

Safety: all public functions swallow their own exceptions and never raise, so the request
control flow that calls record_messages_e2e() is never affected by metrics problems.

Scope/limitation: only the headline endpoint metrics (E2E latency + request count, by
latency_optimisation_flow) are mirrored here. The fine-grained OTel metrics (per-step / LLM /
ttft / tokens) are NOT mirrored — read those via OTLP, or via the single-worker OTel pull
endpoint. This deliberately keeps cardinality low and the change non-invasive.
"""

import os

from letta.log import get_logger

logger = get_logger(__name__)

# Latency buckets in milliseconds, sized for LLM-backed request latencies (tens of ms to ~1 min).
_E2E_MS_BUCKETS = (
    25,
    50,
    100,
    250,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    5000,
    8000,
    13000,
    21000,
    34000,
    60000,
)

_e2e_histogram = None
_request_counter = None
_init_attempted = False


def multiprocess_enabled() -> bool:
    """True when prometheus_client multiprocess mode is configured for this deployment."""
    return bool(os.environ.get("PROMETHEUS_MULTIPROC_DIR"))


def _init_instruments() -> None:
    """Lazily create the native instruments. Idempotent; never raises."""
    global _e2e_histogram, _request_counter, _init_attempted
    if _init_attempted:
        return
    _init_attempted = True
    try:
        from prometheus_client import Counter, Histogram

        _e2e_histogram = Histogram(
            "letta_messages_endpoint_e2e_ms",
            "End-to-end latency (ms) of POST /v1/agents/{id}/messages, partitioned by "
            "latency_optimisation_flow. Native prometheus_client mirror of the OTel "
            "hist_messages_endpoint_e2e_ms; aggregates across uvicorn workers.",
            ["latency_optimisation_flow", "status_code"],
            buckets=_E2E_MS_BUCKETS,
        )
        _request_counter = Counter(
            "letta_messages_endpoint_requests_total",
            "Count of POST /v1/agents/{id}/messages, partitioned by latency_optimisation_flow "
            "and status_code. Native prometheus_client mirror; aggregates across uvicorn workers.",
            ["latency_optimisation_flow", "status_code"],
        )
    except Exception as e:
        logger.warning(f"Native prometheus_client multiprocess instruments unavailable: {e}")


def record_messages_e2e(latency_ms: float, latency_optimisation_flow: str, status_code: int) -> None:
    """Mirror the /messages end-to-end latency onto native prometheus_client instruments.

    No-op unless PROMETHEUS_MULTIPROC_DIR is set. Never raises — safe to call from the
    request hot path alongside the existing OTel recording.
    """
    if not multiprocess_enabled():
        return
    try:
        _init_instruments()
        if _e2e_histogram is None:
            return
        labels = {"latency_optimisation_flow": latency_optimisation_flow, "status_code": str(status_code)}
        _e2e_histogram.labels(**labels).observe(latency_ms)
        _request_counter.labels(**labels).inc()
    except Exception as e:
        logger.warning(f"Failed to record native prometheus messages e2e metric: {e}")


def start_multiprocess_metrics_server(port: int, addr: str) -> bool:
    """Start an HTTP server exposing the multiprocess-aggregated registry on its own port.

    Returns True if this process bound the port. With multiple workers only one wins the
    bind; the rest return False but their data is still served by the winner, because the
    MultiProcessCollector reads every worker's files in PROMETHEUS_MULTIPROC_DIR. Never raises.
    """
    try:
        from prometheus_client import CollectorRegistry, multiprocess, start_http_server

        # Touch the instruments so this process has registered its label files before serving.
        _init_instruments()

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        start_http_server(port=port, addr=addr, registry=registry)
        return True
    except OSError as e:
        # Expected: another worker already bound the port and serves the aggregate for all.
        logger.info(
            f"Multiprocess Prometheus server not bound on {addr}:{port} in this worker ({e}); "
            "another worker serves the aggregated /metrics."
        )
        return False
    except Exception as e:
        logger.warning(f"Failed to start multiprocess Prometheus metrics server on {addr}:{port}: {e}")
        return False
