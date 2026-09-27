from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class StepTimingEntry:
    step_count: int  # agent loop index (0-based)
    step_name: str  # inference type: primary / primary_step0 / haiku_kept / haiku_quality_discarded / primary_after_haiku_quality_retry / primary_after_haiku_timeout
    is_llm_call: bool
    is_success: bool
    provider: str  # model_endpoint_type, e.g. "anthropic_bedrock"
    input_token_count: int
    output_token_count: int
    cache_read_token_count: int
    cache_write_token_count: int
    tool_name: str | None  # tool call returned by LLM; None if text response
    consultant_id: int | None
    user_id: str | None  # self.at_user_id (string in this codebase)
    agent_id: str | None  # set only when both consultant_id and user_id are absent


# Per-request accumulator for agent step inference data.
# Set to a fresh list by RequestLatencyMiddleware at request start;
# LettaAgent._step appends one StepTimingEntry per LLM call (including wasted
# Haiku calls) so the middleware can emit a single structured summary log line
# on completion.
request_step_timings: ContextVar[list[StepTimingEntry] | None] = ContextVar("request_step_timings", default=None)
