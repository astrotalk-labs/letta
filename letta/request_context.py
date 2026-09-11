from contextvars import ContextVar
from typing import Optional

# Per-request accumulator for agent step durations (milliseconds).
# Set to a fresh list by RequestLatencyMiddleware at request start;
# LettaAgent._step appends each step's wall-clock ms so the middleware
# can emit a single structured summary log line on completion.
request_step_timings: ContextVar[Optional[list[int]]] = ContextVar("request_step_timings", default=None)
