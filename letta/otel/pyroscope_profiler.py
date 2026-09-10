"""Grafana Pyroscope continuous profiling for the Letta server.

Optional, opt-in CPU/wall-clock profiling. Enabled via ``settings.pyroscope_enabled``
(``LETTA_PYROSCOPE_ENABLED=true``) and configured through the other ``pyroscope_*`` settings.
Requires the optional ``pyroscope-io`` package; if it is not installed the profiler is skipped with
a warning rather than failing server startup.

Note: the Pyroscope Python SDK profiles CPU/wall-clock time (via py-spy), not heap allocations — it
pinpoints hot code paths, not allocation sites directly. For allocation-level OOM debugging,
complement it with an on-demand tool such as memray or tracemalloc.

This module is the "push" counterpart to go-ai-chat's pprof debug server (`cmd/main/main.go`),
which exposes Go's built-in `net/http/pprof` on an internal port for something external to pull
from. Python has no in-process equivalent (cProfile only profiles the thread that calls it, so an
HTTP handler on its own thread can't attach to the event loop's frames the way pprof attaches
runtime-wide) — the direct analog is `py-spy` itself, run out-of-process against the server's PID
(`py-spy dump --pid 1` / `py-spy record --pid 1 --duration 30 -o profile.svg` via `kubectl exec`),
same as go-ai-chat's "or a manual kubectl port-forward" fallback. `py-spy` is an optional dependency
(extra `profiling`, alongside `pyroscope-io`) so the binary is present in the image; the pod still
needs `CAP_SYS_PTRACE` (or a shared process namespace) granted at deploy time for py-spy to attach.
"""

import os

from letta.log import get_logger

logger = get_logger(__name__)

_is_profiling_initialized = False


def setup_profiling(service_name: str = "letta-server") -> bool:
    """Start Pyroscope continuous profiling if enabled and configured.

    Returns True if profiling was started, False otherwise. Idempotent, and never raises — any
    failure is logged and swallowed so profiling can never take down the server.
    """
    global _is_profiling_initialized

    from letta.settings import settings

    if _is_profiling_initialized:
        return True
    if not settings.pyroscope_enabled:
        return False
    if not settings.pyroscope_server_address:
        logger.warning("[pyroscope] LETTA_PYROSCOPE_ENABLED is set but LETTA_PYROSCOPE_SERVER_ADDRESS is empty; skipping.")
        return False

    try:
        import pyroscope
    except ImportError:
        logger.warning(
            "[pyroscope] pyroscope-io is not installed; skipping continuous profiling. "
            "Install it (poetry add pyroscope-io / pip install pyroscope-io) to enable."
        )
        return False

    # Tags let us filter flame graphs by deployment/pod in Grafana. k8s exposes the pod name via
    # HOSTNAME; ENV_NAME / AWS_REGION mirror the tags used for OTLP tracing.
    tags = {"service_name": service_name}
    env_name = os.getenv("ENV_NAME")
    if env_name:
        tags["env"] = env_name.lower()
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if region:
        tags["region"] = region
    pod = os.getenv("HOSTNAME")
    if pod:
        tags["pod"] = pod

    try:
        pyroscope.configure(
            application_name=service_name,
            server_address=settings.pyroscope_server_address,
            sample_rate=settings.pyroscope_sample_rate,
            detect_subprocesses=True,  # also profile uvicorn worker subprocesses
            oncpu=True,  # on-CPU time only — lower overhead than wall-clock
            gil_only=True,  # only sample threads holding the GIL
            tags=tags,
            basic_auth_username=settings.pyroscope_basic_auth_username or "",  # Grafana Cloud tenant/user
            basic_auth_password=settings.pyroscope_basic_auth_password or "",  # Grafana Cloud API token
        )
    except Exception as e:
        logger.warning(f"[pyroscope] failed to start continuous profiling: {e}")
        return False

    _is_profiling_initialized = True
    logger.info(f"[pyroscope] continuous profiling started: app={service_name} " f"server={settings.pyroscope_server_address} tags={tags}")
    return True
