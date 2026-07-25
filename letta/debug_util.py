import uuid
from collections.abc import Callable
from contextvars import ContextVar

from letta.log import get_logger

logger = get_logger(__name__)
_DEBUG_USER_ID = "125526285"

# Holds a per-request trace ID set at send_message entry.
# Automatically propagated to asyncio tasks via contextvars semantics.
_debug_request_id: ContextVar[str] = ContextVar("debug_request_id", default="")


def set_debug_request_id(user_id, request_id: str) -> None:
    """Set a request ID — no-op for any user other than the debug user."""
    try:
        if str(user_id) == _DEBUG_USER_ID:
            _debug_request_id.set(request_id)
    except Exception:
        pass


def new_debug_request_id(user_id) -> str:
    """Generate and set a short request ID. No-op (returns '') for non-debug users."""
    try:
        if str(user_id) == _DEBUG_USER_ID:
            rid = uuid.uuid4().hex[:12]
            _debug_request_id.set(rid)
            return rid
    except Exception:
        pass
    return ""


def debug_log(user_id, message: str | Callable[[], str]) -> None:
    """Log a debug message only for the hardcoded debug user.

    Pass a callable (lambda) when the message involves expensive computation
    (e.g. json.dumps on large payloads) — it is only evaluated if the user
    matches, and any exception during evaluation or logging is silently swallowed
    so this never causes a regression in the calling code.

    Every log line is automatically prefixed with the current request ID so all
    logs from one end-to-end request can be correlated by filtering on that ID.
    """
    try:
        if str(user_id) == _DEBUG_USER_ID:
            msg = message() if callable(message) else message
            rid = _debug_request_id.get()
            prefix = f"[req={rid}] " if rid else ""
            logger.info("[DEBUG_USER] user_id=%s %s%s", user_id, prefix, msg)
    except Exception:
        pass
