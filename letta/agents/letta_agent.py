import asyncio
import json
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from opentelemetry.trace import Span

from letta.agents.base_agent import BaseAgent
from letta.debug_util import debug_log
from letta.agents.ephemeral_summary_agent import EphemeralSummaryAgent
from letta.agents.helpers import _create_letta_response, _prepare_in_context_messages_no_persist_async, generate_step_id
from letta.constants import DEFAULT_MAX_STEPS
from letta.errors import ContextWindowExceededError
from letta.helpers import ToolRulesSolver
from letta.helpers.datetime_helpers import AsyncTimer, get_utc_timestamp_ns, ns_to_ms
from letta.helpers.tool_execution_helper import enable_strict_mode
from letta.interfaces.anthropic_streaming_interface import AnthropicStreamingInterface
from letta.interfaces.openai_streaming_interface import OpenAIStreamingInterface
from letta.llm_api.llm_client import LLMClient
from letta.llm_api.llm_client_base import LLMClientBase
from letta.local_llm.constants import INNER_THOUGHTS_KWARG
from letta.log import get_logger
from letta.orm.enums import ToolType
from letta.otel.context import get_ctx_attributes
from letta.otel.metric_registry import MetricRegistry
from letta.otel.tracing import log_event, trace_method, tracer
from letta.schemas.agent import AgentState
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message import MessageType
from letta.schemas.letta_message_content import OmittedReasoningContent, ReasoningContent, RedactedReasoningContent, TextContent
from letta.schemas.letta_response import LettaResponse
from letta.schemas.letta_stop_reason import LettaStopReason, StopReasonType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message, MessageCreate
from letta.schemas.openai.chat_completion_response import FunctionCall, ToolCall, UsageStatistics
from letta.schemas.provider_trace import ProviderTraceCreate
from letta.schemas.tool_execution_result import ToolExecutionResult
from letta.schemas.usage import LettaUsageStatistics
from letta.schemas.user import User
from letta.server.rest_api.utils import create_letta_messages_from_llm_response
from letta.services.agent_manager import AgentManager
from letta.services.block_manager import BlockManager
from letta.services.helpers.tool_parser_helper import runtime_override_tool_json_schema
from letta.services.message_manager import MessageManager
from letta.services.passage_manager import PassageManager
from letta.services.step_manager import NoopStepManager, StepManager
from letta.services.summarizer.enums import SummarizationMode
from letta.services.summarizer.summarizer import Summarizer
from letta.services.telemetry_manager import NoopTelemetryManager, TelemetryManager
from letta.services.tool_executor.tool_execution_manager import ToolExecutionManager
from letta.settings import model_settings
from letta.system import package_function_response
from letta.types import JsonDict
from letta.utils import log_telemetry, validate_function_response

logger = get_logger(__name__)


def _log_step_timing(step: str, elapsed_ms: float, **kwargs) -> None:
    """Emit a single structured timing log line with a latency-threshold bracket.

    Brackets:
      [ok]   < 2 s
      [>2s]  >= 2 s and < 5 s
      [>5s]  >= 5 s   ← almost always a problem worth investigating
    """
    if elapsed_ms >= 5_000:
        bracket = ">5s"
    elif elapsed_ms >= 2_000:
        bracket = ">2s"
    else:
        bracket = "ok"
    extra = "  ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
    logger.info("[STEP_TIMING][%s][%s]  duration_ms=%.0f  %s", step, bracket, elapsed_ms, extra)


# Per-request thinking/output_config forwarding is gated to these user IDs.
# Expand once validated. Sonnet 5 is exempt from the gate below since it only
# supports adaptive thinking (no budget_tokens mode), so client-supplied thinking
# must always be honored for that model regardless of user id.
_THINKING_GATED_USER_IDS: frozenset = frozenset({"92744418"})

# Gemini thinking level → approximate token budget
_GEMINI_THINKING_LEVEL_TO_BUDGET: dict = {
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}


def _thinking_override_allowed(at_user_id: Optional[str], model: Optional[str]) -> bool:
    if at_user_id in _THINKING_GATED_USER_IDS:
        return True
    return "sonnet-5" in (model or "").lower()


# Haiku cascade rollout. _CASCADE_ROLLOUT_PCT is the percentage of users (bucketed
# deterministically by user_id % 100) for whom latencyOptimisationFlow is honored.
# Ramp 10 -> 50 -> 100 as production metrics confirm latency improvement; set to 0 for
# an instant full rollback. The always-on test user is honored regardless of the percentage.
_CASCADE_ALWAYS_ON_USER_ID: str = "92744418"
_CASCADE_ROLLOUT_PCT: int = 10


def _cascade_enabled_for_user(at_user_id: Optional[str]) -> bool:
    """Deterministic percentage rollout of the Haiku cascade, keyed on user_id % 100.
    Same user always gets the same treatment. The always-on test user is always enabled;
    missing/non-numeric ids fail closed (cascade off)."""
    if not at_user_id:
        return False
    if _CASCADE_ALWAYS_ON_USER_ID and at_user_id == _CASCADE_ALWAYS_ON_USER_ID:
        return True
    if _CASCADE_ROLLOUT_PCT <= 0:
        return False
    if _CASCADE_ROLLOUT_PCT >= 100:
        return True
    try:
        return (int(at_user_id) % 100) < _CASCADE_ROLLOUT_PCT
    except (ValueError, TypeError):
        return False


# Tools that must always be generated by the primary model (whichever model is configured
# for this request — could be Sonnet, Opus, Vertex variant, etc.; the cascade is model-
# agnostic) — never Haiku. `send_message` is the user-facing reply tool; its text quality
# directly impacts the user experience, so we always re-run on the primary model if Haiku
# picks it. To force any other tool to always use the primary model, add it to this set.
_PRIMARY_RESERVED_TOOLS: frozenset = frozenset(
    {
        "send_message",
    }
)

# Haiku model identifiers per provider, used by the latencyOptimisationFlow cascade.
# Hardcoded for prod stability — Bedrock value is the AstroTalk ap-south-1 application
# inference profile that already points at Claude Haiku 4.5.
_HAIKU_MODEL_ANTHROPIC: str = "claude-haiku-4-5-20251001"
_HAIKU_MODEL_VERTEX: str = "claude-haiku-4-5@20251001"
_HAIKU_MODEL_BEDROCK_ARN: str = (
    # Global system inference profile for Claude Haiku 4.5.
    # Routes worldwide when ap-south-1 is degraded — eliminates the Jun 1 2026
    # incident where the single-region AIP (8y2dovlqcwlc) slowed to 24s/call.
    # Confirmed ACTIVE via list_inference_profiles; latency ~850ms vs ~1240ms.
    "global.anthropic.claude-haiku-4-5-20251001-v1:0"
)

_HAIKU_MODEL_BY_PROVIDER: dict = {
    "anthropic": _HAIKU_MODEL_ANTHROPIC,
    "anthropic_vertex": _HAIKU_MODEL_VERTEX,
    "anthropic_bedrock": _HAIKU_MODEL_BEDROCK_ARN,
}

# Hard cap on how long we'll wait for a Haiku LLM call before falling back to primary.
# When Bedrock degrades (observed: p95 >24s during ap-south-1 service event Jun 2026),
# the timeout bounds the damage to ≤8s per step instead of 24s+ per step.
_HAIKU_TIMEOUT_SECONDS: float = 8.0


class LettaAgent(BaseAgent):

    def __init__(
        self,
        agent_id: str,
        message_manager: MessageManager,
        agent_manager: AgentManager,
        block_manager: BlockManager,
        passage_manager: PassageManager,
        actor: User,
        step_manager: StepManager = NoopStepManager(),
        telemetry_manager: TelemetryManager = NoopTelemetryManager(),
        summary_block_label: str = "conversation_summary",
        message_buffer_limit: int = 60,  # TODO: Make this configurable
        message_buffer_min: int = 15,  # TODO: Make this configurable
        enable_summarization: bool = True,  # TODO: Make this configurable
        max_summarization_retries: int = 3,  # TODO: Make this configurable
        at_user_id: Optional[str] = None,
    ):
        super().__init__(agent_id=agent_id, openai_client=None, message_manager=message_manager, agent_manager=agent_manager, actor=actor)

        # TODO: Make this more general, factorable
        # Summarizer settings
        self.block_manager = block_manager
        self.passage_manager = passage_manager
        self.step_manager = step_manager
        self.telemetry_manager = telemetry_manager
        self.response_messages: List[Message] = []

        self.last_function_response = None

        # Cached archival memory/message size
        self.num_messages = None
        self.num_archival_memories = None

        self.summarization_agent = None
        self.summary_block_label = summary_block_label
        self.max_summarization_retries = max_summarization_retries
        self.at_user_id = at_user_id

        # TODO: Expand to more
        if enable_summarization and (model_settings.letta_embedding_5_4_mini_api_key or model_settings.anthropic_api_key):
            self.summarization_agent = EphemeralSummaryAgent(
                target_block_label=self.summary_block_label,
                agent_id=agent_id,
                block_manager=self.block_manager,
                message_manager=self.message_manager,
                agent_manager=self.agent_manager,
                actor=self.actor,
                at_user_id=at_user_id,
            )

        self.summarizer = Summarizer(
            mode=SummarizationMode.STATIC_MESSAGE_BUFFER,
            summarizer_agent=self.summarization_agent,
            # TODO: Make this configurable
            message_buffer_limit=message_buffer_limit,
            message_buffer_min=message_buffer_min,
        )

    def _apply_provider_switching(
        self,
        agent_state: AgentState,
        use_vertex_experiment: bool,
        use_bedrock_experiment: bool,
        model_override: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ):
        """Apply dynamic provider switching based on experiment flags."""
        # Runtime model override: swap the model name before any endpoint-type remapping
        # so that downstream (e.g. Bedrock ARN selection based on model substring) sees the
        # overridden model.
        if model_override:
            agent_state.llm_config.model = model_override

        # Explicit provider override takes precedence over vertex/bedrock experiment flags.
        # Also auto-detect Gemini models by name so callers don't need to set llm_provider explicitly.
        if llm_provider == "google" or (agent_state.llm_config.model or "").startswith("gemini"):
            agent_state.llm_config.model_endpoint_type = "google_ai"
            return

        original_endpoint_type = agent_state.llm_config.model_endpoint_type
        original_model = agent_state.llm_config.model

        if use_bedrock_experiment and original_endpoint_type == "anthropic":
            agent_state.llm_config.model_endpoint_type = "anthropic_bedrock"
        elif use_vertex_experiment and original_endpoint_type == "anthropic":
            agent_state.llm_config.model_endpoint_type = "anthropic_vertex"
            if "-" in original_model and "@" not in original_model:
                parts = original_model.rsplit("-", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    agent_state.llm_config.model = f"{parts[0]}@{parts[1]}"
        elif not use_vertex_experiment and original_endpoint_type == "anthropic_vertex":
            agent_state.llm_config.model_endpoint_type = "anthropic"
            if "@" in original_model:
                agent_state.llm_config.model = original_model.replace("@", "-")
        elif not use_bedrock_experiment and original_endpoint_type == "anthropic_bedrock":
            agent_state.llm_config.model_endpoint_type = "anthropic"

    @trace_method
    async def step(
        self,
        input_messages: List[MessageCreate],
        max_steps: int = DEFAULT_MAX_STEPS,
        use_assistant_message: bool = True,
        request_start_timestamp_ns: Optional[int] = None,
        include_return_message_types: Optional[List[MessageType]] = None,
        use_vertex_experiment: bool = False,
        use_bedrock_experiment: bool = False,
        model_override: Optional[str] = None,
        user_cohort: Optional[str] = None,
        thinking: Optional[dict] = None,
        thinking_config: Optional[dict] = None,
        output_config: Optional[dict] = None,
        task_id: Optional[str] = None,
        latency_optimisation_flow: bool = False,
        llm_provider: Optional[str] = None,
    ) -> LettaResponse:
        agent_state = await self.agent_manager.get_agent_by_id_async(
            agent_id=self.agent_id, include_relationships=["tools", "memory", "tool_exec_environment_variables"], actor=self.actor
        )
        _, new_in_context_messages, usage, stop_reason = await self._step(
            agent_state=agent_state,
            input_messages=input_messages,
            max_steps=max_steps,
            request_start_timestamp_ns=request_start_timestamp_ns,
            use_vertex_experiment=use_vertex_experiment,
            use_bedrock_experiment=use_bedrock_experiment,
            model_override=model_override,
            user_cohort=user_cohort,
            thinking=thinking,
            thinking_config=thinking_config,
            output_config=output_config,
            task_id=task_id,
            latency_optimisation_flow=latency_optimisation_flow,
            llm_provider=llm_provider,
        )
        return _create_letta_response(
            new_in_context_messages=new_in_context_messages,
            use_assistant_message=use_assistant_message,
            stop_reason=stop_reason,
            usage=usage,
            include_return_message_types=include_return_message_types,
        )

    @trace_method
    async def step_stream_no_tokens(
        self,
        input_messages: List[MessageCreate],
        max_steps: int = DEFAULT_MAX_STEPS,
        use_assistant_message: bool = True,
        request_start_timestamp_ns: Optional[int] = None,
        include_return_message_types: Optional[List[MessageType]] = None,
        use_vertex_experiment: bool = False,
        use_bedrock_experiment: bool = False,
        model_override: Optional[str] = None,
        user_cohort: Optional[str] = None,
        thinking: Optional[dict] = None,
        thinking_config: Optional[dict] = None,
        output_config: Optional[dict] = None,
        llm_provider: Optional[str] = None,
    ):
        agent_state = await self.agent_manager.get_agent_by_id_async(
            agent_id=self.agent_id, include_relationships=["tools", "memory", "tool_exec_environment_variables"], actor=self.actor
        )

        # Handle provider switching based on experiment flags
        self._apply_provider_switching(
            agent_state, use_vertex_experiment, use_bedrock_experiment, model_override=model_override, llm_provider=llm_provider
        )

        async with AsyncTimer() as _t:
            current_in_context_messages, new_in_context_messages = await _prepare_in_context_messages_no_persist_async(
                input_messages, agent_state, self.message_manager, self.actor
            )
        _log_step_timing(
            "message_buffer_load",
            _t.elapsed_ms,
            agent_id=agent_state.id,
            user_id=self.at_user_id,
            msg_count=len(current_in_context_messages),
        )
        initial_messages = new_in_context_messages
        tool_rules_solver = ToolRulesSolver(agent_state.tool_rules)
        llm_client = LLMClient.create(
            provider_type=agent_state.llm_config.model_endpoint_type,
            put_inner_thoughts_first=True,
            actor=self.actor,
            at_user_id=self.at_user_id,
            user_cohort=user_cohort,
        )
        stop_reason = None
        usage = LettaUsageStatistics()

        # span for request
        request_span = tracer.start_span("time_to_first_token", start_time=request_start_timestamp_ns)
        request_span.set_attributes({f"llm_config.{k}": v for k, v in agent_state.llm_config.model_dump().items() if v is not None})

        _chain_at_uid = getattr(self, "at_user_id", None)
        for i in range(max_steps):
            step_id = generate_step_id()
            step_start = get_utc_timestamp_ns()
            agent_step_span = tracer.start_span("agent_step", start_time=step_start)
            agent_step_span.set_attributes({"step_id": step_id})

            try:
                from letta.llm_api.anthropic_client import _is_user_in_cache_obs_sample
                import json as _json

                if _chain_at_uid and _is_user_in_cache_obs_sample(_chain_at_uid):
                    logger.info(
                        "[CHAIN_OBS] %s",
                        _json.dumps(
                            {"phase": "step_start", "at_user_id": _chain_at_uid, "agent_id": agent_state.id, "step_index": i}, default=str
                        ),
                    )
            except Exception:
                pass
            debug_log(
                self.at_user_id,
                f"step_stream_no_tokens: LOOP step={i}/{max_steps} agent_id={agent_state.id} model={agent_state.llm_config.model}",
            )

            request_data, response_data, current_in_context_messages, new_in_context_messages, valid_tool_names = (
                await self._build_and_request_from_llm(
                    current_in_context_messages,
                    new_in_context_messages,
                    agent_state,
                    llm_client,
                    tool_rules_solver,
                    agent_step_span,
                    use_vertex_experiment=use_vertex_experiment,
                    use_bedrock_experiment=use_bedrock_experiment,
                    step_index=i,
                    thinking=thinking,
                    thinking_config=thinking_config,
                    output_config=output_config,
                )
            )
            in_context_messages = current_in_context_messages + new_in_context_messages

            log_event("agent.stream_no_tokens.llm_response.received")  # [3^]

            response = llm_client.convert_response_to_chat_completion(response_data, in_context_messages, agent_state.llm_config)

            # update usage
            # TODO: add run_id
            usage.step_count += 1
            usage.completion_tokens += response.usage.completion_tokens
            usage.prompt_tokens += response.usage.prompt_tokens
            usage.total_tokens += response.usage.total_tokens
            MetricRegistry().message_output_tokens.record(
                response.usage.completion_tokens, dict(get_ctx_attributes(), **{"model.name": agent_state.llm_config.model})
            )

            if not response.choices[0].message.tool_calls:
                text = response.choices[0].message.content
                if text:
                    synthetic = ToolCall(
                        id=f"synthetic_{uuid.uuid4().hex[:8]}",
                        function=FunctionCall(
                            name="send_message",
                            arguments=json.dumps({"message": text}),
                        ),
                    )
                    response.choices[0].message.tool_calls = [synthetic]
                    logger.warning("Model returned text without a tool call; wrapping as synthetic send_message.")
                else:
                    raise ValueError("No tool calls found in response, model must make a tool call")
            tool_call = response.choices[0].message.tool_calls[0]
            if response.choices[0].message.reasoning_content:
                reasoning = [
                    ReasoningContent(
                        reasoning=response.choices[0].message.reasoning_content,
                        is_native=True,
                        signature=response.choices[0].message.reasoning_content_signature,
                    )
                ]
            elif response.choices[0].message.omitted_reasoning_content:
                reasoning = [OmittedReasoningContent()]
            elif response.choices[0].message.content:
                reasoning = [TextContent(text=response.choices[0].message.content)]  # reasoning placed into content for legacy reasons
            else:
                logger.info("No reasoning content found.")
                reasoning = None

            persisted_messages, should_continue, stop_reason = await self._handle_ai_response(
                tool_call,
                valid_tool_names,
                agent_state,
                tool_rules_solver,
                response.usage,
                reasoning_content=reasoning,
                initial_messages=initial_messages,
                agent_step_span=agent_step_span,
                is_final_step=(i == max_steps - 1),
            )
            self.response_messages.extend(persisted_messages)
            new_in_context_messages.extend(persisted_messages)
            initial_messages = None
            log_event("agent.stream_no_tokens.llm_response.processed")  # [4^]

            # log step time
            now = get_utc_timestamp_ns()
            step_ns = now - step_start
            agent_step_span.add_event(name="step_ms", attributes={"duration_ms": ns_to_ms(step_ns)})
            agent_step_span.end()

            # Log LLM Trace
            async with AsyncTimer() as _t:
                await self.telemetry_manager.create_provider_trace_async(
                    actor=self.actor,
                    provider_trace_create=ProviderTraceCreate(
                        request_json=request_data,
                        response_json=response_data,
                        step_id=step_id,
                        organization_id=self.actor.organization_id,
                    ),
                )
            _log_step_timing("telemetry_persist", _t.elapsed_ms, agent_id=agent_state.id, user_id=self.at_user_id, step_id=step_id)

            # stream step
            # TODO: improve TTFT
            filter_user_messages = [m for m in persisted_messages if m.role != "user"]
            letta_messages = Message.to_letta_messages_from_list(
                filter_user_messages, use_assistant_message=use_assistant_message, reverse=False
            )

            for message in letta_messages:
                if include_return_message_types is None or message.message_type in include_return_message_types:
                    yield f"data: {message.model_dump_json()}\n\n"

            MetricRegistry().step_execution_time_ms_histogram.record(step_start - get_utc_timestamp_ns(), get_ctx_attributes())

            if i == max_steps - 1 and should_continue:
                try:
                    import json as _json

                    logger.warning(
                        "[COST_LEAK] %s",
                        _json.dumps(
                            {"type": "max_steps_hit", "at_user_id": _chain_at_uid, "agent_id": agent_state.id, "step_count": i + 1},
                            default=str,
                        ),
                    )
                except Exception:
                    pass

            if not should_continue:
                break

        try:
            from letta.llm_api.anthropic_client import _is_user_in_cache_obs_sample
            import json as _json

            if _chain_at_uid and _is_user_in_cache_obs_sample(_chain_at_uid):
                logger.info(
                    "[CHAIN_OBS] %s",
                    _json.dumps(
                        {
                            "phase": "turn_complete",
                            "at_user_id": _chain_at_uid,
                            "agent_id": agent_state.id,
                            "total_steps": usage.step_count,
                            "output_tokens": usage.completion_tokens,
                            "input_tokens": usage.prompt_tokens,
                        },
                        default=str,
                    ),
                )
        except Exception:
            pass

        # Extend the in context message ids
        if not agent_state.message_buffer_autoclear:
            await self._rebuild_context_window(
                in_context_messages=current_in_context_messages,
                new_letta_messages=new_in_context_messages,
                llm_config=agent_state.llm_config,
                total_tokens=usage.total_tokens,
                force=False,
            )

        # log request time
        if request_start_timestamp_ns:
            now = get_utc_timestamp_ns()
            request_ns = now - request_start_timestamp_ns
            request_span.add_event(name="letta_request_ms", attributes={"duration_ms": ns_to_ms(request_ns)})
        request_span.end()

        # Return back usage
        for finish_chunk in self.get_finish_chunks_for_stream(usage, stop_reason):
            yield f"data: {finish_chunk}\n\n"

    async def _step(
        self,
        agent_state: AgentState,
        input_messages: List[MessageCreate],
        max_steps: int = DEFAULT_MAX_STEPS,
        request_start_timestamp_ns: Optional[int] = None,
        use_vertex_experiment: bool = False,
        use_bedrock_experiment: bool = False,
        model_override: Optional[str] = None,
        user_cohort: Optional[str] = None,
        thinking: Optional[dict] = None,
        thinking_config: Optional[dict] = None,
        output_config: Optional[dict] = None,
        task_id: Optional[str] = None,
        latency_optimisation_flow: bool = False,
        llm_provider: Optional[str] = None,
    ) -> Tuple[List[Message], List[Message], Optional[LettaStopReason], LettaUsageStatistics]:
        """
        Carries out an invocation of the agent loop. In each step, the agent
            1. Rebuilds its memory
            2. Generates a request for the LLM
            3. Fetches a response from the LLM
            4. Processes the response
        """
        # Handle provider switching based on experiment flags
        self._apply_provider_switching(
            agent_state, use_vertex_experiment, use_bedrock_experiment, model_override=model_override, llm_provider=llm_provider
        )

        # Stash task_id on the instance so agent-internal helpers (_rebuild_memory_async,
        # _rebuild_context_window) can tag their latency logs. Agent is per-request, so safe.
        self._task_id = task_id

        # Cascade rollout gate: honor latencyOptimisationFlow only for the rolled-out
        # percentage of users (_CASCADE_ROLLOUT_PCT, keyed on user_id % 100). Ramp up as
        # metrics confirm improvement; set the pct to 0 for an instant rollback.
        if latency_optimisation_flow and not _cascade_enabled_for_user(self.at_user_id):
            logger.info(
                f"[HAIKU_CASCADE] task_id={task_id or 'N/A'} GATED_OFF "
                f"user_id={self.at_user_id} rollout_pct={_CASCADE_ROLLOUT_PCT} — running primary-only"
            )
            debug_log(self.at_user_id, f"_step: cascade GATED_OFF rollout_pct={_CASCADE_ROLLOUT_PCT} → latency_optimisation_flow=False")
            latency_optimisation_flow = False
        else:
            debug_log(self.at_user_id, f"_step: cascade gate passed latency_optimisation_flow={latency_optimisation_flow}")

        _ctx_prep_start = get_utc_timestamp_ns() if task_id else None
        async with AsyncTimer() as _t:
            current_in_context_messages, new_in_context_messages = await _prepare_in_context_messages_no_persist_async(
                input_messages, agent_state, self.message_manager, self.actor
            )
        if task_id and _ctx_prep_start:
            logger.warning(
                f"[TASK_LATENCY] task_id={task_id} phase=context_prep duration_ms={ns_to_ms(get_utc_timestamp_ns() - _ctx_prep_start)}"
            )
        _log_step_timing(
            "message_buffer_load",
            _t.elapsed_ms,
            agent_id=agent_state.id,
            user_id=self.at_user_id,
            msg_count=len(current_in_context_messages),
        )
        initial_messages = new_in_context_messages
        tool_rules_solver = ToolRulesSolver(agent_state.tool_rules)
        llm_client = LLMClient.create(
            provider_type=agent_state.llm_config.model_endpoint_type,
            put_inner_thoughts_first=True,
            actor=self.actor,
            at_user_id=self.at_user_id,
            user_cohort=user_cohort,
        )

        # Resolve the Haiku model name for this provider (used when latency_optimisation_flow=True).
        # Hardcoded provider→model mapping; Bedrock uses an ARN application inference profile.
        _haiku_model: Optional[str] = None

        # Pre-build attribute dicts for the cascade metrics. Low cardinality only —
        # provider + primary_model + (outcome on the step counter). Wrapped in a
        # local helper so we never silently mutate the request context attributes.
        def _cascade_attrs(**extra) -> dict:
            base = dict(get_ctx_attributes())
            base["provider"] = agent_state.llm_config.model_endpoint_type
            base["primary_model"] = agent_state.llm_config.model
            base.update(extra)
            return base

        if latency_optimisation_flow:
            # Guard: if the effective primary model is already a Haiku variant the
            # cascade would route Haiku→Haiku, wasting tokens with no latency benefit.
            # This can happen when model_override passes a Haiku model name or ARN.
            if "haiku" in (agent_state.llm_config.model or "").lower():
                logger.warning(
                    f"[HAIKU_CASCADE] task_id={task_id or 'N/A'} DISABLED_PRIMARY_IS_HAIKU "
                    f"primary_model={agent_state.llm_config.model} — "
                    f"cascade skipped (Haiku→Haiku is a no-op)"
                )
                MetricRegistry().haiku_cascade_request_counter.add(1, _cascade_attrs(state="disabled_primary_is_haiku"))
                debug_log(self.at_user_id, f"_step: cascade DISABLED_PRIMARY_IS_HAIKU model={agent_state.llm_config.model}")
                latency_optimisation_flow = False
            else:
                _haiku_model = _HAIKU_MODEL_BY_PROVIDER.get(agent_state.llm_config.model_endpoint_type)
                if _haiku_model:
                    logger.warning(
                        f"[HAIKU_CASCADE] task_id={task_id or 'N/A'} ENABLED "
                        f"provider={agent_state.llm_config.model_endpoint_type} "
                        f"primary_model={agent_state.llm_config.model} "
                        f"haiku_model={_haiku_model} "
                        f"rule=step0_on_primary_then_tool_calls_on_haiku_send_message_never_on_haiku "
                        f"primary_reserved_tools={sorted(_PRIMARY_RESERVED_TOOLS)} "
                        f"strategy=skip_step0_plus_post_call_gate_only_plus_timeout "
                        f"timeout_s={_HAIKU_TIMEOUT_SECONDS}"
                    )
                    MetricRegistry().haiku_cascade_request_counter.add(1, _cascade_attrs(state="enabled"))
                    debug_log(
                        self.at_user_id, f"_step: cascade ENABLED haiku_model={_haiku_model} primary_model={agent_state.llm_config.model}"
                    )
                else:
                    logger.warning(
                        f"[HAIKU_CASCADE] task_id={task_id or 'N/A'} DISABLED_NO_MODEL "
                        f"latencyOptimisationFlow=True but no Haiku model mapped "
                        f"for provider={agent_state.llm_config.model_endpoint_type}; using primary model for all steps."
                    )
                    MetricRegistry().haiku_cascade_request_counter.add(1, _cascade_attrs(state="disabled_no_model"))
                    debug_log(self.at_user_id, f"_step: cascade DISABLED_NO_MODEL provider={agent_state.llm_config.model_endpoint_type}")
        else:
            debug_log(
                self.at_user_id,
                f"_step: latency_optimisation_flow=False using primary model for all steps model={agent_state.llm_config.model}",
            )

        # span for request
        request_span = tracer.start_span("time_to_first_token")
        request_span.set_attributes({f"llm_config.{k}": v for k, v in agent_state.llm_config.model_dump().items() if v is not None})

        stop_reason = None
        usage = LettaUsageStatistics()
        _prev_tool_name: Optional[str] = None  # tracks the tool called in the previous step for cascade decisions

        # Cascade usage counters (only meaningful when latency_optimisation_flow=True)
        _haiku_step_count: int = 0
        _primary_step_count: int = 0
        _haiku_timeout_step_count: int = 0  # steps where Haiku timed out → fell back to primary
        _haiku_prompt_tokens: int = 0
        _haiku_completion_tokens: int = 0
        _primary_prompt_tokens: int = 0
        _primary_completion_tokens: int = 0
        # Wasted Haiku calls — tokens from cascade attempts we discarded because Haiku
        # picked a non-memory tool (we re-ran on the primary model to preserve quality).
        _haiku_retried_step_count: int = 0
        _haiku_wasted_prompt_tokens: int = 0
        _haiku_wasted_completion_tokens: int = 0

        _cascade_chain_at_uid = getattr(self, "at_user_id", None)
        for i in range(max_steps):
            step_id = generate_step_id()
            step_start = get_utc_timestamp_ns()
            agent_step_span = tracer.start_span("agent_step", start_time=step_start)
            agent_step_span.set_attributes({"step_id": step_id})

            try:
                from letta.llm_api.anthropic_client import _is_user_in_cache_obs_sample
                import json as _json

                if _cascade_chain_at_uid and _is_user_in_cache_obs_sample(_cascade_chain_at_uid):
                    logger.info(
                        "[CHAIN_OBS] %s",
                        _json.dumps(
                            {"phase": "step_start", "at_user_id": _cascade_chain_at_uid, "agent_id": agent_state.id, "step_index": i},
                            default=str,
                        ),
                    )
            except Exception:
                pass
            debug_log(self.at_user_id, f"_step: LOOP step={i}/{max_steps} agent_id={agent_state.id} model={agent_state.llm_config.model}")

            # Cascade rule: attempt Haiku from step 1 onwards.
            #
            # Why we DON'T attempt Haiku at step 0:
            #   For short user messages the agent often goes straight to send_message
            #   in a single step. Attempting Haiku for that step:
            #     - costs the full system prompt as uncached Haiku tokens (~6k for
            #       this agent), and
            #     - the quality gate must then retry on primary anyway because Haiku
            #       either returns text or hallucinates send_message.
            #   Production traces showed ~6k wasted Haiku tokens + ~3s extra latency
            #   per single-step request, with zero compensating savings.
            #
            # For multi-step flows the savings come from steps i>0: those are tool
            # calls that Haiku handles well and where prompt caching keeps the
            # follow-up Haiku tokens cheap. So skip step 0, run step 1+ on Haiku.
            # The quality gate still catches Haiku picking send_message and retries
            # on primary, preserving user-facing response quality.
            #
            # Snapshot the pre-call message state so we can re-run cleanly if the gate fires.
            _pre_call_current_messages = current_in_context_messages
            _pre_call_new_messages = new_in_context_messages

            _original_model: Optional[str] = None
            _step_used_haiku: bool = False
            _step_was_retried: bool = False  # flipped to True if the gate forces a primary retry
            if _haiku_model and i > 0:
                _original_model = agent_state.llm_config.model
                agent_state.llm_config.model = _haiku_model
                _step_used_haiku = True
                logger.info(
                    f"[HAIKU_CASCADE] task_id={task_id or 'N/A'} step={i} ATTEMPT_HAIKU " f"model={_original_model} -> {_haiku_model}"
                )
                debug_log(self.at_user_id, f"_step: step={i} ATTEMPT_HAIKU primary={_original_model} → haiku={_haiku_model}")
            elif _haiku_model and i == 0:
                logger.warning(
                    f"[HAIKU_CASCADE] task_id={task_id or 'N/A'} step={i} KEEP_PRIMARY "
                    f"reason=step_0_always_uses_primary_to_avoid_single_step_request_tax"
                )
                MetricRegistry().haiku_cascade_step_counter.add(1, _cascade_attrs(outcome="primary_step0"))
                debug_log(self.at_user_id, f"_step: step={i} KEEP_PRIMARY model={agent_state.llm_config.model}")
            elif not _haiku_model and latency_optimisation_flow:
                # Cascade requested but no Haiku model mapped — record so we can detect misconfigs.
                MetricRegistry().haiku_cascade_step_counter.add(1, _cascade_attrs(outcome="primary_no_haiku_model"))
                debug_log(self.at_user_id, f"_step: step={i} no haiku model mapped using primary={agent_state.llm_config.model}")
            else:
                debug_log(self.at_user_id, f"_step: step={i} standard (no cascade) model={agent_state.llm_config.model}")

            # Strategy: gate-only (NOT tool-list restriction).
            #
            # Haiku gets the FULL tool list (including send_message). The post-call gate
            # below enforces "send_message runs on primary": if Haiku picks send_message,
            # we discard and re-run the step on the primary model.
            #
            # We do NOT strip send_message from Haiku. Stripping it removed the only
            # trigger for the primary handoff (the gate only fires when Haiku picks
            # send_message), so the agent could never terminate on a Haiku step and looped
            # to max_steps (prod task 323737721: 42+ archival_memory_insert steps, ~150s).
            _excluded_for_haiku: Optional[frozenset] = None

            _llm_start = get_utc_timestamp_ns() if task_id else None
            _haiku_timed_out: bool = False
            response = None  # always initialised — assigned in try block or timeout fallback
            try:
                _haiku_coro = self._build_and_request_from_llm(
                    current_in_context_messages,
                    new_in_context_messages,
                    agent_state,
                    llm_client,
                    tool_rules_solver,
                    agent_step_span,
                    use_vertex_experiment=use_vertex_experiment,
                    use_bedrock_experiment=use_bedrock_experiment,
                    step_index=i,
                    thinking=thinking,
                    thinking_config=thinking_config,
                    output_config=output_config,
                    excluded_tool_names=_excluded_for_haiku,
                )
                # Timeout circuit-breaker: if Haiku is slower than _HAIKU_TIMEOUT_SECONDS
                # (Bedrock degradation), fall through to primary instead of blocking.
                if _step_used_haiku:
                    try:
                        result = await asyncio.wait_for(_haiku_coro, timeout=_HAIKU_TIMEOUT_SECONDS)
                    except asyncio.TimeoutError:
                        _haiku_timed_out = True
                        _elapsed_ms = ns_to_ms(get_utc_timestamp_ns() - _llm_start) if _llm_start else 0
                        logger.warning(
                            f"[HAIKU_CASCADE] task_id={task_id or 'N/A'} step={i} TIMEOUT_SKIP_TO_PRIMARY "
                            f"elapsed_ms={_elapsed_ms} timeout_s={_HAIKU_TIMEOUT_SECONDS} "
                            f"— falling back to primary model"
                        )
                        MetricRegistry().haiku_cascade_step_counter.add(1, _cascade_attrs(outcome="primary_haiku_timeout"))
                        result = None
                else:
                    result = await _haiku_coro

                if result is not None:
                    request_data, response_data, current_in_context_messages, new_in_context_messages, valid_tool_names = result
                    if task_id and _llm_start:
                        logger.warning(
                            f"[TASK_LATENCY] task_id={task_id} step={i} phase=llm_call duration_ms={ns_to_ms(get_utc_timestamp_ns() - _llm_start)}"
                        )
                    in_context_messages = current_in_context_messages + new_in_context_messages
                    log_event("agent.step.llm_response.received")  # [3^]
                    response = llm_client.convert_response_to_chat_completion(response_data, in_context_messages, agent_state.llm_config)
            finally:
                # Always restore the primary model — even if the LLM call raises.
                if _original_model is not None:
                    agent_state.llm_config.model = _original_model
                    _original_model = None

            # Haiku timed out → run on primary immediately (no gate needed).
            if _haiku_timed_out:
                _haiku_timeout_step_count += 1
                _step_used_haiku = False
                debug_log(self.at_user_id, f"_step: step={i} HAIKU_TIMED_OUT retrying on primary model={agent_state.llm_config.model}")
                _llm_timeout_retry_start = get_utc_timestamp_ns() if task_id else None
                request_data, response_data, current_in_context_messages, new_in_context_messages, valid_tool_names = (
                    await self._build_and_request_from_llm(
                        _pre_call_current_messages,
                        _pre_call_new_messages,
                        agent_state,
                        llm_client,
                        tool_rules_solver,
                        agent_step_span,
                        use_vertex_experiment=use_vertex_experiment,
                        use_bedrock_experiment=use_bedrock_experiment,
                        step_index=i,
                        thinking=thinking,
                        thinking_config=thinking_config,
                        output_config=output_config,
                    )
                )
                if task_id and _llm_timeout_retry_start:
                    logger.warning(
                        f"[TASK_LATENCY] task_id={task_id} step={i} phase=llm_call_timeout_fallback "
                        f"duration_ms={ns_to_ms(get_utc_timestamp_ns() - _llm_timeout_retry_start)}"
                    )
                in_context_messages = current_in_context_messages + new_in_context_messages
                response = llm_client.convert_response_to_chat_completion(response_data, in_context_messages, agent_state.llm_config)

            # Safety guard — response must be assigned by this point.
            if response is None:
                raise RuntimeError(
                    f"[HAIKU_CASCADE] task_id={task_id or 'N/A'} step={i}: "
                    f"response is None after LLM call — haiku_timed_out={_haiku_timed_out}"
                )

            # Quality gate — the sole enforcement of "send_message never runs on Haiku".
            #
            # Haiku gets the full tool list and tool_choice="any" (forced tool call), so
            # text responses should not occur. If Haiku does pick send_message (or, in
            # the unlikely event it returns text with no tool call), we discard that
            # response and re-run the step on the primary model so the user-facing
            # reply is always generated by the configured primary model.
            if _step_used_haiku:
                _haiku_tool_call_name: Optional[str] = (
                    response.choices[0].message.tool_calls[0].function.name if response.choices[0].message.tool_calls else None
                )
                # Retry when Haiku picked a primary-reserved tool (send_message) or
                # unexpectedly returned no tool call at all.
                _needs_primary_retry = _haiku_tool_call_name in _PRIMARY_RESERVED_TOOLS or _haiku_tool_call_name is None
                debug_log(
                    self.at_user_id, f"_step: step={i} quality gate haiku_tool={_haiku_tool_call_name} needs_retry={_needs_primary_retry}"
                )
                if _needs_primary_retry:
                    _reason = (
                        "haiku_returned_text_response_no_tool"
                        if _haiku_tool_call_name is None
                        else f"haiku_picked_primary_reserved_tool_{_haiku_tool_call_name}"
                    )
                    logger.warning(
                        f"[HAIKU_CASCADE] task_id={task_id or 'N/A'} step={i} RETRY_ON_PRIMARY "
                        f"haiku_picked={_haiku_tool_call_name or 'text_response'} "
                        f"reason={_reason} "
                        f"wasted_tokens={response.usage.total_tokens}"
                    )
                    # Track wasted Haiku usage before we discard the response
                    _haiku_retried_step_count += 1
                    _haiku_wasted_prompt_tokens += response.usage.prompt_tokens
                    _haiku_wasted_completion_tokens += response.usage.completion_tokens

                    # OTel: count this retry and record how many tokens we wasted.
                    _retry_attrs = _cascade_attrs(
                        outcome="primary_retried",
                        retry_reason=("text_response" if _haiku_tool_call_name is None else "primary_reserved_tool"),
                    )
                    MetricRegistry().haiku_cascade_step_counter.add(1, _retry_attrs)
                    MetricRegistry().haiku_cascade_wasted_tokens_histogram.record(response.usage.total_tokens, _retry_attrs)
                    _step_was_retried = True

                    # Re-run on the primary model.
                    # The model is already restored above; the pre-call message snapshot avoids
                    # double-processing the in-context history the discarded call mutated.
                    _llm_retry_start = get_utc_timestamp_ns() if task_id else None
                    request_data, response_data, current_in_context_messages, new_in_context_messages, valid_tool_names = (
                        await self._build_and_request_from_llm(
                            _pre_call_current_messages,
                            _pre_call_new_messages,
                            agent_state,
                            llm_client,
                            tool_rules_solver,
                            agent_step_span,
                            use_vertex_experiment=use_vertex_experiment,
                            use_bedrock_experiment=use_bedrock_experiment,
                            step_index=i,
                            thinking=thinking,
                            thinking_config=thinking_config,
                            output_config=output_config,
                        )
                    )
                    if task_id and _llm_retry_start:
                        logger.warning(
                            f"[TASK_LATENCY] task_id={task_id} step={i} phase=llm_call_retry "
                            f"duration_ms={ns_to_ms(get_utc_timestamp_ns() - _llm_retry_start)}"
                        )
                    in_context_messages = current_in_context_messages + new_in_context_messages
                    response = llm_client.convert_response_to_chat_completion(response_data, in_context_messages, agent_state.llm_config)
                    # This step ultimately ran on the primary model, not Haiku
                    _step_used_haiku = False
                    debug_log(
                        self.at_user_id,
                        lambda _i=i, _r=response: f"_step: step={_i} PRIMARY_RETRY complete tool={_r.choices[0].message.tool_calls[0].function.name if _r.choices[0].message.tool_calls else 'text'}",
                    )

            # TODO: add run_id
            usage.step_count += 1
            usage.completion_tokens += response.usage.completion_tokens
            usage.prompt_tokens += response.usage.prompt_tokens
            usage.total_tokens += response.usage.total_tokens
            MetricRegistry().message_output_tokens.record(
                response.usage.completion_tokens, dict(get_ctx_attributes(), **{"model.name": agent_state.llm_config.model})
            )

            # Accumulate per-model usage for the cascade summary
            if _step_used_haiku:
                _haiku_step_count += 1
                _haiku_prompt_tokens += response.usage.prompt_tokens
                _haiku_completion_tokens += response.usage.completion_tokens
            else:
                _primary_step_count += 1
                _primary_prompt_tokens += response.usage.prompt_tokens
                _primary_completion_tokens += response.usage.completion_tokens

            if not response.choices[0].message.tool_calls:
                text = response.choices[0].message.content
                if text:
                    synthetic = ToolCall(
                        id=f"synthetic_{uuid.uuid4().hex[:8]}",
                        function=FunctionCall(
                            name="send_message",
                            arguments=json.dumps({"message": text}),
                        ),
                    )
                    response.choices[0].message.tool_calls = [synthetic]
                    logger.warning("Model returned text without a tool call; wrapping as synthetic send_message.")
                else:
                    raise ValueError("No tool calls found in response, model must make a tool call")
            tool_call = response.choices[0].message.tool_calls[0]
            if response.choices[0].message.reasoning_content:
                reasoning = [
                    ReasoningContent(
                        reasoning=response.choices[0].message.reasoning_content,
                        is_native=True,
                        signature=response.choices[0].message.reasoning_content_signature,
                    )
                ]
            elif response.choices[0].message.content:
                reasoning = [TextContent(text=response.choices[0].message.content)]  # reasoning placed into content for legacy reasons
            elif response.choices[0].message.omitted_reasoning_content:
                reasoning = [OmittedReasoningContent()]
            else:
                logger.info("No reasoning content found.")
                reasoning = None

            _tool_start = get_utc_timestamp_ns() if task_id else None
            persisted_messages, should_continue, stop_reason = await self._handle_ai_response(
                tool_call,
                valid_tool_names,
                agent_state,
                tool_rules_solver,
                response.usage,
                reasoning_content=reasoning,
                step_id=step_id,
                initial_messages=initial_messages,
                agent_step_span=agent_step_span,
                is_final_step=(i == max_steps - 1),
            )
            if task_id and _tool_start:
                logger.warning(
                    f"[TASK_LATENCY] task_id={task_id} step={i} phase=tool_exec duration_ms={ns_to_ms(get_utc_timestamp_ns() - _tool_start)}"
                )
            self.response_messages.extend(persisted_messages)
            new_in_context_messages.extend(persisted_messages)
            initial_messages = None
            _prev_tool_name = tool_call.function.name  # used by next iteration for cascade decision

            # Per-step usage log — useful when latency_optimisation_flow is on to confirm which
            # model handled each step and how many tokens it cost.
            if latency_optimisation_flow:
                logger.warning(
                    f"[HAIKU_CASCADE] task_id={task_id or 'N/A'} step={i} USAGE "
                    f"used_haiku={_step_used_haiku} "
                    f"model={_haiku_model if _step_used_haiku else agent_state.llm_config.model} "
                    f"tool={_prev_tool_name} "
                    f"prompt_tokens={response.usage.prompt_tokens} "
                    f"completion_tokens={response.usage.completion_tokens} "
                    f"total_tokens={response.usage.total_tokens}"
                )
                # OTel: count this step's resolution. If this step ran Haiku (and the gate
                # didn't fire), `outcome=haiku_kept`. If it ran primary because of step 0
                # the primary_step0 counter was already added above; here we only emit
                # primary_kept for steps that ran primary for some other reason (e.g. the
                # cascade was disabled mid-run). The retry path emits its own outcome above.
                if _step_used_haiku:
                    MetricRegistry().haiku_cascade_step_counter.add(1, _cascade_attrs(outcome="haiku_kept", tool=_prev_tool_name))

            log_event("agent.step.llm_response.processed")  # [4^]

            # log step time
            now = get_utc_timestamp_ns()
            step_ns = now - step_start
            agent_step_span.add_event(name="step_ms", attributes={"duration_ms": ns_to_ms(step_ns)})
            if task_id:
                logger.warning(f"[TASK_LATENCY] task_id={task_id} step={i} phase=total_step duration_ms={ns_to_ms(step_ns)}")

            # OTel: per-step latency tagged with cascade outcome. Lets us answer
            # "how long do Haiku-kept steps take vs primary-retried steps?".
            if latency_optimisation_flow:
                if _step_used_haiku:
                    _step_outcome = "haiku_kept"
                elif _step_was_retried:
                    _step_outcome = "primary_retried"
                elif _haiku_model and i == 0:
                    _step_outcome = "primary_step0"
                elif not _haiku_model:
                    _step_outcome = "primary_no_haiku_model"
                else:
                    _step_outcome = "primary_kept"
                MetricRegistry().haiku_cascade_step_ms_histogram.record(ns_to_ms(step_ns), _cascade_attrs(outcome=_step_outcome))

            agent_step_span.end()

            # Log LLM Trace
            async with AsyncTimer() as _t:
                await self.telemetry_manager.create_provider_trace_async(
                    actor=self.actor,
                    provider_trace_create=ProviderTraceCreate(
                        request_json=request_data,
                        response_json=response_data,
                        step_id=step_id,
                        organization_id=self.actor.organization_id,
                    ),
                )
            _log_step_timing("telemetry_persist", _t.elapsed_ms, agent_id=agent_state.id, user_id=self.at_user_id, step_id=step_id)

            MetricRegistry().step_execution_time_ms_histogram.record(step_start - get_utc_timestamp_ns(), get_ctx_attributes())

            if i == max_steps - 1 and should_continue:
                try:
                    import json as _json

                    logger.warning(
                        "[COST_LEAK] %s",
                        _json.dumps(
                            {"type": "max_steps_hit", "at_user_id": _cascade_chain_at_uid, "agent_id": agent_state.id, "step_count": i + 1},
                            default=str,
                        ),
                    )
                except Exception:
                    pass

            if not should_continue:
                break

        if _haiku_retried_step_count > 0:
            try:
                import json as _json

                logger.warning(
                    "[COST_LEAK] %s",
                    _json.dumps(
                        {
                            "type": "haiku_cascade_retry",
                            "at_user_id": _cascade_chain_at_uid,
                            "agent_id": agent_state.id,
                            "retry_count": _haiku_retried_step_count,
                            "primary_steps": _primary_step_count,
                            "haiku_steps": _haiku_step_count,
                        },
                        default=str,
                    ),
                )
            except Exception:
                pass

        try:
            from letta.llm_api.anthropic_client import _is_user_in_cache_obs_sample
            import json as _json

            if _cascade_chain_at_uid and _is_user_in_cache_obs_sample(_cascade_chain_at_uid):
                logger.info(
                    "[CHAIN_OBS] %s",
                    _json.dumps(
                        {
                            "phase": "turn_complete",
                            "at_user_id": _cascade_chain_at_uid,
                            "agent_id": agent_state.id,
                            "total_steps": usage.step_count,
                            "output_tokens": usage.completion_tokens,
                            "input_tokens": usage.prompt_tokens,
                        },
                        default=str,
                    ),
                )
        except Exception:
            pass

        # log request time
        if request_start_timestamp_ns:
            now = get_utc_timestamp_ns()
            request_ns = now - request_start_timestamp_ns
            request_span.add_event(name="request_ms", attributes={"duration_ms": ns_to_ms(request_ns)})
        request_span.end()

        # Extend the in context message ids
        _ctx_rebuild_start = get_utc_timestamp_ns() if task_id else None
        if not agent_state.message_buffer_autoclear:
            await self._rebuild_context_window(
                in_context_messages=current_in_context_messages,
                new_letta_messages=new_in_context_messages,
                llm_config=agent_state.llm_config,
                total_tokens=usage.total_tokens,
                force=False,
            )
        if task_id and _ctx_rebuild_start:
            logger.warning(
                f"[TASK_LATENCY] task_id={task_id} phase=ctx_rebuild duration_ms={ns_to_ms(get_utc_timestamp_ns() - _ctx_rebuild_start)}"
            )
        if task_id and request_start_timestamp_ns:
            logger.warning(
                f"[TASK_LATENCY] task_id={task_id} phase=total_request duration_ms={ns_to_ms(get_utc_timestamp_ns() - request_start_timestamp_ns)}"
            )

        # Cascade SUMMARY — fires whenever the flag was requested, even if the cascade was
        # disabled mid-flight (so we can see "asked for it but got zero haiku steps" cases).
        if latency_optimisation_flow:
            _total_steps = _haiku_step_count + _primary_step_count
            _haiku_total_tokens = _haiku_prompt_tokens + _haiku_completion_tokens
            _primary_total_tokens = _primary_prompt_tokens + _primary_completion_tokens
            _haiku_wasted_total_tokens = _haiku_wasted_prompt_tokens + _haiku_wasted_completion_tokens
            _haiku_step_pct = (100.0 * _haiku_step_count / _total_steps) if _total_steps else 0.0
            logger.warning(
                f"[HAIKU_CASCADE] task_id={task_id or 'N/A'} SUMMARY "
                f"total_steps={_total_steps} "
                f"haiku_steps={_haiku_step_count} primary_steps={_primary_step_count} "
                f"haiku_pct={_haiku_step_pct:.1f} "
                f"haiku_tokens={_haiku_total_tokens} (prompt={_haiku_prompt_tokens} completion={_haiku_completion_tokens}) "
                f"primary_tokens={_primary_total_tokens} (prompt={_primary_prompt_tokens} completion={_primary_completion_tokens}) "
                f"retried_steps={_haiku_retried_step_count} "
                f"timeout_steps={_haiku_timeout_step_count} "
                f"wasted_haiku_tokens={_haiku_wasted_total_tokens} (prompt={_haiku_wasted_prompt_tokens} completion={_haiku_wasted_completion_tokens}) "
                f"haiku_model={_haiku_model or 'N/A'}"
            )

            # OTel: per-request histograms so the cascade is observable in Grafana
            # without scraping log lines. `had_retry` lets the request counter slice
            # be partitioned by whether a retry fired. `total_steps_bucket` bins
            # the step count so dashboards can split p50/p95 latency by workload
            # shape without needing a heatmap.
            def _bucket(n: int) -> str:
                if n <= 1:
                    return "1"
                if n == 2:
                    return "2"
                if n <= 5:
                    return "3-5"
                return "6+"

            _summary_attrs = _cascade_attrs(
                had_retry=str(_haiku_retried_step_count > 0).lower(),
                total_steps_bucket=_bucket(_total_steps),
            )
            MetricRegistry().haiku_cascade_total_steps_histogram.record(_total_steps, _summary_attrs)
            MetricRegistry().haiku_cascade_haiku_pct_histogram.record(_haiku_step_pct, _summary_attrs)
            MetricRegistry().haiku_cascade_haiku_tokens_histogram.record(_haiku_total_tokens, _summary_attrs)
            MetricRegistry().haiku_cascade_primary_tokens_histogram.record(_primary_total_tokens, _summary_attrs)

            # The headline cascade metric: total end-to-end latency for cascade-enabled
            # requests, partitioned by had_retry and total_steps_bucket. Use this to
            # prove "the cascade made things faster" by comparing percentiles vs the
            # same time window before the cascade went live.
            if request_start_timestamp_ns:
                _request_ms = ns_to_ms(get_utc_timestamp_ns() - request_start_timestamp_ns)
                MetricRegistry().haiku_cascade_request_ms_histogram.record(_request_ms, _summary_attrs)

        return current_in_context_messages, new_in_context_messages, usage, stop_reason

    @trace_method
    async def step_stream(
        self,
        input_messages: List[MessageCreate],
        max_steps: int = DEFAULT_MAX_STEPS,
        use_assistant_message: bool = True,
        request_start_timestamp_ns: Optional[int] = None,
        include_return_message_types: Optional[List[MessageType]] = None,
        use_vertex_experiment: bool = False,
        use_bedrock_experiment: bool = False,
        model_override: Optional[str] = None,
        user_cohort: Optional[str] = None,
        thinking: Optional[dict] = None,
        thinking_config: Optional[dict] = None,
        output_config: Optional[dict] = None,
        llm_provider: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Carries out an invocation of the agent loop in a streaming fashion that yields partial tokens.
        Whenever we detect a tool call, we yield from _handle_ai_response as well. At each step, the agent
            1. Rebuilds its memory
            2. Generates a request for the LLM
            3. Fetches a response from the LLM
            4. Processes the response
        """
        agent_state = await self.agent_manager.get_agent_by_id_async(
            agent_id=self.agent_id, include_relationships=["tools", "memory", "tool_exec_environment_variables"], actor=self.actor
        )

        # Handle provider switching based on experiment flags
        self._apply_provider_switching(
            agent_state, use_vertex_experiment, use_bedrock_experiment, model_override=model_override, llm_provider=llm_provider
        )

        current_in_context_messages, new_in_context_messages = await _prepare_in_context_messages_no_persist_async(
            input_messages, agent_state, self.message_manager, self.actor
        )
        initial_messages = new_in_context_messages

        tool_rules_solver = ToolRulesSolver(agent_state.tool_rules)
        llm_client = LLMClient.create(
            provider_type=agent_state.llm_config.model_endpoint_type,
            put_inner_thoughts_first=True,
            actor=self.actor,
            at_user_id=self.at_user_id,
            user_cohort=user_cohort,
        )
        stop_reason = None
        usage = LettaUsageStatistics()
        first_chunk, request_span = True, None
        if request_start_timestamp_ns:
            request_span = tracer.start_span("time_to_first_token", start_time=request_start_timestamp_ns)
            request_span.set_attributes({f"llm_config.{k}": v for k, v in agent_state.llm_config.model_dump().items() if v is not None})

        for i in range(max_steps):
            step_id = generate_step_id()
            step_start = get_utc_timestamp_ns()
            agent_step_span = tracer.start_span("agent_step", start_time=step_start)
            agent_step_span.set_attributes({"step_id": step_id})

            (
                request_data,
                stream,
                current_in_context_messages,
                new_in_context_messages,
                valid_tool_names,
                provider_request_start_timestamp_ns,
            ) = await self._build_and_request_from_llm_streaming(
                first_chunk,
                agent_step_span,
                request_start_timestamp_ns,
                current_in_context_messages,
                new_in_context_messages,
                agent_state,
                llm_client,
                tool_rules_solver,
                step_index=i,
                thinking=thinking,
                output_config=output_config,
            )
            log_event("agent.stream.llm_response.received")  # [3^]

            # TODO: THIS IS INCREDIBLY UGLY
            # TODO: THERE ARE MULTIPLE COPIES OF THE LLM_CONFIG EVERYWHERE THAT ARE GETTING MANIPULATED
            if agent_state.llm_config.model_endpoint_type in ("anthropic", "anthropic_vertex", "anthropic_bedrock"):
                interface = AnthropicStreamingInterface(
                    use_assistant_message=use_assistant_message,
                    put_inner_thoughts_in_kwarg=agent_state.llm_config.put_inner_thoughts_in_kwargs,
                )
            elif agent_state.llm_config.model_endpoint_type == "openai":
                interface = OpenAIStreamingInterface(
                    use_assistant_message=use_assistant_message,
                    put_inner_thoughts_in_kwarg=agent_state.llm_config.put_inner_thoughts_in_kwargs,
                )
            else:
                raise ValueError(f"Streaming not supported for {agent_state.llm_config}")

            async for chunk in interface.process(
                stream, ttft_span=request_span, provider_request_start_timestamp_ns=provider_request_start_timestamp_ns
            ):
                # Measure time to first token
                if first_chunk and request_span is not None:
                    now = get_utc_timestamp_ns()
                    ttft_ns = now - request_start_timestamp_ns
                    request_span.add_event(name="time_to_first_token_ms", attributes={"ttft_ms": ns_to_ms(ttft_ns)})
                    metric_attributes = get_ctx_attributes()
                    metric_attributes["model.name"] = agent_state.llm_config.model
                    MetricRegistry().ttft_ms_histogram.record(ns_to_ms(ttft_ns), metric_attributes)
                    first_chunk = False

                if include_return_message_types is None or chunk.message_type in include_return_message_types:
                    # filter down returned data
                    yield f"data: {chunk.model_dump_json()}\n\n"

            stream_end_time_ns = get_utc_timestamp_ns()

            # update usage
            usage.step_count += 1
            usage.completion_tokens += interface.output_tokens
            usage.prompt_tokens += interface.input_tokens
            usage.total_tokens += interface.input_tokens + interface.output_tokens
            MetricRegistry().message_output_tokens.record(
                interface.output_tokens, dict(get_ctx_attributes(), **{"model.name": agent_state.llm_config.model})
            )

            # log LLM request time
            llm_request_ms = ns_to_ms(stream_end_time_ns - request_start_timestamp_ns)
            agent_step_span.add_event(name="llm_request_ms", attributes={"duration_ms": llm_request_ms})
            MetricRegistry().llm_execution_time_ms_histogram.record(
                llm_request_ms,
                dict(get_ctx_attributes(), **{"model.name": agent_state.llm_config.model}),
            )
            # Per-LLM-call latency, partitioned by Bedrock vs non-Bedrock provider.
            MetricRegistry().llm_call_ms_histogram.record(
                llm_request_ms,
                dict(
                    get_ctx_attributes(),
                    **{
                        "model.name": agent_state.llm_config.model,
                        "use_bedrock_experiment": str(use_bedrock_experiment).lower(),
                    },
                ),
            )

            # Process resulting stream content
            try:
                tool_call = interface.get_tool_call_object()
            except ValueError as e:
                stop_reason = LettaStopReason(stop_reason=StopReasonType.no_tool_call.value)
                yield f"data: {stop_reason.model_dump_json()}\n\n"
                raise e
            except Exception as e:
                stop_reason = LettaStopReason(stop_reason=StopReasonType.invalid_tool_call.value)
                yield f"data: {stop_reason.model_dump_json()}\n\n"
                raise e
            reasoning_content = interface.get_reasoning_content()
            persisted_messages, should_continue, stop_reason = await self._handle_ai_response(
                tool_call,
                valid_tool_names,
                agent_state,
                tool_rules_solver,
                UsageStatistics(
                    completion_tokens=interface.output_tokens,
                    prompt_tokens=interface.input_tokens,
                    total_tokens=interface.input_tokens + interface.output_tokens,
                ),
                reasoning_content=reasoning_content,
                pre_computed_assistant_message_id=interface.letta_message_id,
                step_id=step_id,
                initial_messages=initial_messages,
                agent_step_span=agent_step_span,
                is_final_step=(i == max_steps - 1),
            )
            self.response_messages.extend(persisted_messages)
            new_in_context_messages.extend(persisted_messages)
            initial_messages = None

            # log total step time
            now = get_utc_timestamp_ns()
            step_ns = now - step_start
            agent_step_span.add_event(name="step_ms", attributes={"duration_ms": ns_to_ms(step_ns)})
            agent_step_span.end()

            # TODO (cliandy): the stream POST request span has ended at this point, we should tie this to the stream
            # log_event("agent.stream.llm_response.processed") # [4^]

            # Log LLM Trace
            # TODO (cliandy): we are piecing together the streamed response here. Content here does not match the actual response schema.
            await self.telemetry_manager.create_provider_trace_async(
                actor=self.actor,
                provider_trace_create=ProviderTraceCreate(
                    request_json=request_data,
                    response_json={
                        "content": {
                            "tool_call": tool_call.model_dump_json(),
                            "reasoning": [content.model_dump_json() for content in reasoning_content],
                        },
                        "id": interface.message_id,
                        "model": interface.model,
                        "role": "assistant",
                        # "stop_reason": "",
                        # "stop_sequence": None,
                        "type": "message",
                        "usage": {"input_tokens": interface.input_tokens, "output_tokens": interface.output_tokens},
                    },
                    step_id=step_id,
                    organization_id=self.actor.organization_id,
                ),
            )

            tool_return = [msg for msg in persisted_messages if msg.role == "tool"][-1].to_letta_messages()[0]
            if not (use_assistant_message and tool_return.name == "send_message"):
                # Apply message type filtering if specified
                if include_return_message_types is None or tool_return.message_type in include_return_message_types:
                    yield f"data: {tool_return.model_dump_json()}\n\n"

            # TODO (cliandy): consolidate and expand with trace
            MetricRegistry().step_execution_time_ms_histogram.record(step_start - get_utc_timestamp_ns(), get_ctx_attributes())

            if not should_continue:
                break

        # Extend the in context message ids
        if not agent_state.message_buffer_autoclear:
            await self._rebuild_context_window(
                in_context_messages=current_in_context_messages,
                new_letta_messages=new_in_context_messages,
                llm_config=agent_state.llm_config,
                total_tokens=usage.total_tokens,
                force=False,
            )

        # log time of entire request
        if request_start_timestamp_ns:
            now = get_utc_timestamp_ns()
            request_ns = now - request_start_timestamp_ns
            request_span.add_event(name="letta_request_ms", attributes={"duration_ms": ns_to_ms(request_ns)})
        request_span.end()

        for finish_chunk in self.get_finish_chunks_for_stream(usage, stop_reason):
            yield f"data: {finish_chunk}\n\n"

    # noinspection PyInconsistentReturns
    async def _build_and_request_from_llm(
        self,
        current_in_context_messages: List[Message],
        new_in_context_messages: List[Message],
        agent_state: AgentState,
        llm_client: LLMClientBase,
        tool_rules_solver: ToolRulesSolver,
        agent_step_span: "Span",
        use_vertex_experiment: bool = False,
        use_bedrock_experiment: bool = False,
        step_index: int = 0,
        thinking: Optional[dict] = None,
        thinking_config: Optional[dict] = None,
        output_config: Optional[dict] = None,
        excluded_tool_names: Optional[frozenset] = None,
    ) -> Tuple[Dict, Dict, List[Message], List[Message], List[str]] | None:
        for attempt in range(self.max_summarization_retries + 1):
            try:
                log_event("agent.stream_no_tokens.messages.refreshed")
                # Create LLM request data
                async with AsyncTimer() as _t:
                    request_data, valid_tool_names = await self._create_llm_request_data_async(
                        llm_client=llm_client,
                        in_context_messages=current_in_context_messages + new_in_context_messages,
                        agent_state=agent_state,
                        tool_rules_solver=tool_rules_solver,
                        step_index=step_index,
                    )
                _log_step_timing(
                    "llm_request_build",
                    _t.elapsed_ms,
                    agent_id=agent_state.id,
                    user_id=self.at_user_id,
                    step_idx=step_index,
                    model=agent_state.llm_config.model,
                )
                log_event("agent.stream_no_tokens.llm_request.created")

                # Haiku tool restriction: strip primary-reserved tools so Haiku cannot
                # pick send_message directly. Keep tool_choice="any" (NOT "auto") so
                # Haiku must still call *some* tool.
                # Guard: skip if filtering would empty the list or break a forced call.
                if excluded_tool_names and request_data.get("tools"):
                    _filtered_tools = [t for t in request_data["tools"] if t.get("name") not in excluded_tool_names]
                    _tc_name = (request_data.get("tool_choice") or {}).get("name")
                    _forced_call_stripped = bool(_tc_name and _tc_name in excluded_tool_names)
                    if _filtered_tools and not _forced_call_stripped:
                        request_data["tools"] = _filtered_tools
                        valid_tool_names = [n for n in valid_tool_names if n not in excluded_tool_names]
                        debug_log(
                            self.at_user_id,
                            f"_build_and_request: step={step_index} tool restriction applied excluded={excluded_tool_names} remaining={[t.get('name') for t in _filtered_tools]}",
                        )
                    else:
                        logger.warning(
                            f"[HAIKU_CASCADE] step={step_index} SKIP_TOOL_RESTRICTION "
                            f"reason={'empty_after_filter' if not _filtered_tools else 'forced_call_stripped'} "
                            f"tool_choice_name={_tc_name} — gate will handle"
                        )
                        debug_log(
                            self.at_user_id,
                            f"_build_and_request: step={step_index} SKIP_TOOL_RESTRICTION reason={'empty_after_filter' if not _filtered_tools else 'forced_call_stripped'}",
                        )
                else:
                    debug_log(
                        self.at_user_id,
                        f"_build_and_request: step={step_index} no tool restriction excluded_tool_names={excluded_tool_names}",
                    )

                # Inject per-request thinking overrides (gated, except Sonnet 5 which is always exempt)
                if _thinking_override_allowed(self.at_user_id, agent_state.llm_config.model):
                    debug_log(
                        self.at_user_id,
                        f"_build_and_request: step={step_index} thinking override ALLOWED model={agent_state.llm_config.model} thinking={thinking}",
                    )
                    if thinking is not None:
                        request_data["thinking"] = thinking
                        request_data["temperature"] = 1.0
                        if request_data.get("max_tokens", 0) < 8000:
                            request_data["max_tokens"] = 8000
                        # Thinking is incompatible with tool_choice "any"/"tool"; downgrade to "auto"
                        if request_data.get("tool_choice", {}).get("type") in ("any", "tool"):
                            request_data["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
                        debug_log(self.at_user_id, f"_build_and_request: step={step_index} thinking injected into request_data")
                    else:
                        debug_log(
                            self.at_user_id, f"_build_and_request: step={step_index} thinking override allowed but thinking=None skipping"
                        )
                else:
                    debug_log(
                        self.at_user_id,
                        f"_build_and_request: step={step_index} thinking override NOT allowed model={agent_state.llm_config.model}",
                    )
                # output_config flows to all users unconditionally
                if output_config is not None:
                    request_data["output_config"] = output_config
                    debug_log(self.at_user_id, f"_build_and_request: step={step_index} output_config injected={output_config}")
                else:
                    debug_log(self.at_user_id, f"_build_and_request: step={step_index} output_config=None skipping")
                # Inject Gemini thinking_config: maps {"level": "low"|"medium"|"high"} → thinking_budget
                if thinking_config is not None and agent_state.llm_config.model_endpoint_type in ("google_ai", "google_vertex"):
                    level = thinking_config.get("level", "low")
                    budget = _GEMINI_THINKING_LEVEL_TO_BUDGET.get(level, _GEMINI_THINKING_LEVEL_TO_BUDGET["low"])
                    if "config" in request_data:
                        request_data["config"]["thinking_config"] = {"thinking_budget": budget}
                    debug_log(
                        self.at_user_id,
                        f"_build_and_request: step={step_index} gemini thinking_config injected level={level} budget={budget}",
                    )
                else:
                    debug_log(
                        self.at_user_id,
                        f"_build_and_request: step={step_index} gemini thinking_config skipped thinking_config={thinking_config} endpoint_type={agent_state.llm_config.model_endpoint_type}",
                    )

                if agent_state.llm_config.model_endpoint_type in ("google_ai", "google_vertex"):
                    logger.warning(
                        f"[GEMINI_REQUEST] at_user_id={self.at_user_id} model={agent_state.llm_config.model} "
                        f"endpoint_type={agent_state.llm_config.model_endpoint_type} "
                        f"thinking_config_in={thinking_config} "
                        f"thinking_config_sent={request_data.get('config', {}).get('thinking_config')} "
                        f"temperature={request_data.get('config', {}).get('temperature')} "
                        f"max_output_tokens={request_data.get('config', {}).get('max_output_tokens')}"
                    )

                debug_log(
                    self.at_user_id,
                    lambda _si=step_index, _rd=request_data: f"_build_and_request: step={_si} FINAL_LLM_PROMPT model={agent_state.llm_config.model} endpoint_type={agent_state.llm_config.model_endpoint_type} prompt={json.dumps(_rd, default=str)}",
                )
                async with AsyncTimer() as timer:
                    # Attempt LLM request
                    response = await llm_client.request_async(
                        request_data,
                        agent_state.llm_config,
                        use_vertex_experiment=use_vertex_experiment,
                        use_bedrock_experiment=use_bedrock_experiment,
                    )
                MetricRegistry().llm_execution_time_ms_histogram.record(
                    timer.elapsed_ms,
                    dict(get_ctx_attributes(), **{"model.name": agent_state.llm_config.model}),
                )
                # Per-LLM-call latency, partitioned by Bedrock vs non-Bedrock provider.
                MetricRegistry().llm_call_ms_histogram.record(
                    timer.elapsed_ms,
                    dict(
                        get_ctx_attributes(),
                        **{
                            "model.name": agent_state.llm_config.model,
                            "use_bedrock_experiment": str(use_bedrock_experiment).lower(),
                        },
                    ),
                )
                agent_step_span.add_event(name="llm_request_ms", attributes={"duration_ms": timer.elapsed_ms})
                _log_step_timing(
                    "llm_call",
                    timer.elapsed_ms,
                    agent_id=agent_state.id,
                    user_id=self.at_user_id,
                    step_idx=step_index,
                    model=agent_state.llm_config.model,
                    provider=agent_state.llm_config.model_endpoint_type,
                )

                return request_data, response, current_in_context_messages, new_in_context_messages, valid_tool_names

            except Exception as e:
                if attempt == self.max_summarization_retries:
                    raise e

                # Handle the error and prepare for retry
                current_in_context_messages = await self._handle_llm_error(
                    e,
                    llm_client=llm_client,
                    in_context_messages=current_in_context_messages,
                    new_letta_messages=new_in_context_messages,
                    llm_config=agent_state.llm_config,
                    force=True,
                )
                new_in_context_messages = []
                log_event(f"agent.stream_no_tokens.retry_attempt.{attempt + 1}")

    # noinspection PyInconsistentReturns
    async def _build_and_request_from_llm_streaming(
        self,
        first_chunk: bool,
        ttft_span: "Span",
        request_start_timestamp_ns: int,
        current_in_context_messages: List[Message],
        new_in_context_messages: List[Message],
        agent_state: AgentState,
        llm_client: LLMClientBase,
        tool_rules_solver: ToolRulesSolver,
        step_index: int = 0,
        thinking: Optional[dict] = None,
        output_config: Optional[dict] = None,
    ) -> Tuple[Dict, AsyncStream[ChatCompletionChunk], List[Message], List[Message], List[str], int] | None:
        for attempt in range(self.max_summarization_retries + 1):
            try:
                log_event("agent.stream_no_tokens.messages.refreshed")
                # Create LLM request data
                request_data, valid_tool_names = await self._create_llm_request_data_async(
                    llm_client=llm_client,
                    in_context_messages=current_in_context_messages + new_in_context_messages,
                    agent_state=agent_state,
                    tool_rules_solver=tool_rules_solver,
                    step_index=step_index,
                )
                log_event("agent.stream.llm_request.created")  # [2^]

                # Inject per-request thinking overrides (gated, except Sonnet 5 which is always exempt)
                if _thinking_override_allowed(self.at_user_id, agent_state.llm_config.model):
                    if thinking is not None:
                        request_data["thinking"] = thinking
                        request_data["temperature"] = 1.0
                        if request_data.get("max_tokens", 0) < 8000:
                            request_data["max_tokens"] = 8000
                        # Thinking is incompatible with tool_choice "any"/"tool"; downgrade to "auto"
                        if request_data.get("tool_choice", {}).get("type") in ("any", "tool"):
                            request_data["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
                # output_config flows to all users unconditionally
                if output_config is not None:
                    request_data["output_config"] = output_config

                provider_request_start_timestamp_ns = get_utc_timestamp_ns()
                if first_chunk and ttft_span is not None:
                    request_start_to_provider_request_start_ns = provider_request_start_timestamp_ns - request_start_timestamp_ns
                    ttft_span.add_event(
                        name="request_start_to_provider_request_start_ns",
                        attributes={"request_start_to_provider_request_start_ns": ns_to_ms(request_start_to_provider_request_start_ns)},
                    )

                # Attempt LLM request
                return (
                    request_data,
                    await llm_client.stream_async(request_data, agent_state.llm_config),
                    current_in_context_messages,
                    new_in_context_messages,
                    valid_tool_names,
                    provider_request_start_timestamp_ns,
                )

            except Exception as e:
                if attempt == self.max_summarization_retries:
                    raise e

                # Handle the error and prepare for retry
                current_in_context_messages = await self._handle_llm_error(
                    e,
                    llm_client=llm_client,
                    in_context_messages=current_in_context_messages,
                    new_letta_messages=new_in_context_messages,
                    llm_config=agent_state.llm_config,
                    force=True,
                )
                new_in_context_messages: list[Message] = []
                log_event(f"agent.stream_no_tokens.retry_attempt.{attempt + 1}")

    @trace_method
    async def _handle_llm_error(
        self,
        e: Exception,
        llm_client: LLMClientBase,
        in_context_messages: List[Message],
        new_letta_messages: List[Message],
        llm_config: LLMConfig,
        force: bool,
    ) -> List[Message]:
        debug_log(
            self.at_user_id,
            f"_handle_llm_error: error_type={type(e).__name__} is_context_window_exceeded={isinstance(e, ContextWindowExceededError)} force={force} in_context_msg_count={len(in_context_messages)}",
        )
        if isinstance(e, ContextWindowExceededError):
            debug_log(
                self.at_user_id,
                f"_handle_llm_error: CONTEXT_WINDOW_EXCEEDED → triggering _rebuild_context_window model={llm_config.model} context_window={llm_config.context_window}",
            )
            return await self._rebuild_context_window(
                in_context_messages=in_context_messages, new_letta_messages=new_letta_messages, llm_config=llm_config, force=force
            )
        else:
            debug_log(self.at_user_id, f"_handle_llm_error: non-context-window error → re-raising error_type={type(e).__name__}")
            raise llm_client.handle_llm_error(e)

    @trace_method
    async def _rebuild_context_window(
        self,
        in_context_messages: List[Message],
        new_letta_messages: List[Message],
        llm_config: LLMConfig,
        total_tokens: Optional[int] = None,
        force: bool = False,
    ) -> List[Message]:
        # If total tokens is reached, we truncate down
        # TODO: This can be broken by bad configs, e.g. lower bound too high, initial messages too fat, etc.
        debug_log(
            self.at_user_id,
            f"_rebuild_context_window: entry force={force} total_tokens={total_tokens} context_window={llm_config.context_window} in_context_msg_count={len(in_context_messages)}",
        )
        if force or (total_tokens and total_tokens > llm_config.context_window):
            self.logger.warning(
                f"Total tokens {total_tokens} exceeds configured max tokens {llm_config.context_window}, forcefully clearing message history."
            )
            debug_log(
                self.at_user_id,
                f"_rebuild_context_window: FORCE_CLEAR path total_tokens={total_tokens} context_window={llm_config.context_window}",
            )
            new_in_context_messages, updated = self.summarizer.summarize(
                in_context_messages=in_context_messages, new_letta_messages=new_letta_messages, force=True, clear=True
            )
        else:
            debug_log(self.at_user_id, f"_rebuild_context_window: SOFT_SUMMARIZE path")
            new_in_context_messages, updated = self.summarizer.summarize(
                in_context_messages=in_context_messages, new_letta_messages=new_letta_messages
            )
        debug_log(
            self.at_user_id, f"_rebuild_context_window: summarize complete updated={updated} new_msg_count={len(new_in_context_messages)}"
        )
        if updated:
            try:
                import json as _json

                logger.warning(
                    "[COST_LEAK] %s",
                    _json.dumps(
                        {
                            "type": "summarization_triggered",
                            "at_user_id": getattr(self, "at_user_id", None),
                            "agent_id": getattr(self, "agent_id", None),
                        },
                        default=str,
                    ),
                )
            except Exception:
                pass
        _t_setctx = get_utc_timestamp_ns()
        await self.agent_manager.set_in_context_messages_async(
            agent_id=self.agent_id, message_ids=[m.id for m in new_in_context_messages], actor=self.actor
        )
        if self._task_id:
            logger.info(
                f"[CTXWIN_TIMING] task_id={self._task_id} agent_id={self.agent_id} "
                f"set_in_context_ms={ns_to_ms(get_utc_timestamp_ns() - _t_setctx)} summarized={updated}"
            )

        return new_in_context_messages

    @trace_method
    async def summarize_conversation_history(self) -> AgentState:
        agent_state = await self.agent_manager.get_agent_by_id_async(agent_id=self.agent_id, actor=self.actor)
        message_ids = agent_state.message_ids
        in_context_messages = await self.message_manager.get_messages_by_ids_async(message_ids=message_ids, actor=self.actor)
        new_in_context_messages, updated = self.summarizer.summarize(
            in_context_messages=in_context_messages, new_letta_messages=[], force=True
        )
        return await self.agent_manager.set_in_context_messages_async(
            agent_id=self.agent_id, message_ids=[m.id for m in new_in_context_messages], actor=self.actor
        )

    @trace_method
    async def _create_llm_request_data_async(
        self,
        llm_client: LLMClientBase,
        in_context_messages: List[Message],
        agent_state: AgentState,
        tool_rules_solver: ToolRulesSolver,
        step_index: int = 0,
    ) -> Tuple[dict, List[str]]:
        self.num_messages, self.num_archival_memories = await asyncio.gather(
            (
                self.message_manager.size_async(actor=self.actor, agent_id=agent_state.id)
                if self.num_messages is None
                else asyncio.sleep(0, result=self.num_messages)
            ),
            (
                self.passage_manager.agent_passage_size_async(actor=self.actor, agent_id=agent_state.id)
                if self.num_archival_memories is None
                else asyncio.sleep(0, result=self.num_archival_memories)
            ),
        )

        self._current_step_index = step_index
        try:
            from letta.llm_api.anthropic_client import _is_user_in_cache_obs_sample
            import json as _json

            _chain_uid = getattr(self, "at_user_id", None)
            if _chain_uid and _is_user_in_cache_obs_sample(_chain_uid):
                logger.info(
                    "[CHAIN_OBS] %s",
                    _json.dumps(
                        {
                            "phase": "llm_request_build",
                            "at_user_id": _chain_uid,
                            "agent_id": agent_state.id,
                            "step_index": step_index,
                            "num_messages": self.num_messages,
                            "num_archival_memories": self.num_archival_memories,
                        },
                        default=str,
                    ),
                )
        except Exception:
            pass

        # PR β cascade fix: skip _rebuild_memory_async on intermediate steps of the
        # same user turn. The agent multi-step loop calls core_memory_append between
        # LLM calls, which used to trigger a rebuild every step and invalidate the
        # system-prompt cache.
        #
        # ROLLED BACK to test-user-only after universal graduation (PR #77) showed
        # cost-per-minute +2.9% vs pre-graduation in a clean traffic-normalized
        # comparison (₹3.96/min → ₹4.07/min, 20h post-grad). The reason: the
        # tool-executor path (core_memory_append → update_memory_if_changed_async
        # → rebuild_system_prompt_async) rebuilds system prompt DURING tool
        # execution, independent of this code path. Skipping the agent-loop
        # rebuild without also gating the tool-executor rebuild causes a second-
        # order effect where the next user turn's rebuild has a bigger
        # accumulated diff, hurting cache more on subsequent turns.
        #
        # Restoring test-user gate while we ship the tool-executor cascade fix
        # (next PR). Will re-graduate this together with that fix once both
        # paths skip in coordinated fashion.
        skip_rebuild = self.at_user_id == "92744418" and step_index > 0
        if skip_rebuild:
            try:
                import json as _json

                logger.info(
                    "[REBUILD_SKIPPED_MID_TURN] %s",
                    _json.dumps(
                        {
                            "at_user_id": self.at_user_id,
                            "agent_id": agent_state.id,
                            "step_index": step_index,
                        },
                        default=str,
                    ),
                )
            except Exception:
                pass
        else:
            async with AsyncTimer() as _t:
                in_context_messages = await self._rebuild_memory_async(
                    in_context_messages,
                    agent_state,
                    num_messages=self.num_messages,
                    num_archival_memories=self.num_archival_memories,
                    tool_rules_solver=tool_rules_solver,
                )
            _log_step_timing(
                "memory_rebuild",
                _t.elapsed_ms,
                agent_id=agent_state.id,
                user_id=self.at_user_id,
                step_idx=step_index,
                archival_count=self.num_archival_memories,
                msg_count=self.num_messages,
            )

        tools = [
            t
            for t in agent_state.tools
            if t.tool_type
            in {
                ToolType.CUSTOM,
                ToolType.LETTA_CORE,
                ToolType.LETTA_MEMORY_CORE,
                ToolType.LETTA_MULTI_AGENT_CORE,
                ToolType.LETTA_SLEEPTIME_CORE,
                ToolType.LETTA_BUILTIN,
                ToolType.LETTA_FILES_CORE,
                ToolType.EXTERNAL_COMPOSIO,
                ToolType.EXTERNAL_MCP,
            }
        ]

        # Mirror the sync agent loop: get allowed tools or allow all if none are allowed
        self.last_function_response = self._load_last_function_response(in_context_messages)
        valid_tool_names = tool_rules_solver.get_allowed_tool_names(
            available_tools=set([t.name for t in tools]),
            last_function_response=self.last_function_response,
        ) or list(set(t.name for t in tools))

        # TODO: Copied from legacy agent loop, so please be cautious
        # Set force tool
        force_tool_call = None
        if len(valid_tool_names) == 1:
            force_tool_call = valid_tool_names[0]

        allowed_tools = [enable_strict_mode(t.json_schema) for t in tools if t.name in set(valid_tool_names)]
        allowed_tools = runtime_override_tool_json_schema(
            tool_list=allowed_tools, response_format=agent_state.response_format, request_heartbeat=True
        )

        return (
            llm_client.build_request_data(
                in_context_messages,
                agent_state.llm_config,
                allowed_tools,
                force_tool_call,
            ),
            valid_tool_names,
        )

    @trace_method
    async def _handle_ai_response(
        self,
        tool_call: ToolCall,
        valid_tool_names: List[str],
        agent_state: AgentState,
        tool_rules_solver: ToolRulesSolver,
        usage: UsageStatistics,
        reasoning_content: Optional[List[Union[TextContent, ReasoningContent, RedactedReasoningContent, OmittedReasoningContent]]] = None,
        pre_computed_assistant_message_id: Optional[str] = None,
        step_id: str | None = None,
        initial_messages: Optional[List[Message]] = None,
        agent_step_span: Optional["Span"] = None,
        is_final_step: Optional[bool] = None,
    ) -> Tuple[List[Message], bool, Optional[LettaStopReason]]:
        """
        Now that streaming is done, handle the final AI response.
        This might yield additional SSE tokens if we do stalling.
        At the end, set self._continue_execution accordingly.
        """
        stop_reason = None
        # Check if the called tool is allowed by tool name:
        tool_call_name = tool_call.function.name
        tool_call_args_str = tool_call.function.arguments

        # Temp hack to gracefully handle parallel tool calling attempt, only take first one
        if "}{" in tool_call_args_str:
            tool_call_args_str = tool_call_args_str.split("}{", 1)[0] + "}"

        try:
            tool_args = json.loads(tool_call_args_str)
            assert isinstance(tool_args, dict), "tool_args must be a dict"
        except json.JSONDecodeError:
            tool_args = {}
        except AssertionError:
            tool_args = json.loads(tool_args)

        if is_final_step:
            stop_reason = LettaStopReason(stop_reason=StopReasonType.max_steps.value)
            logger.info("Agent has reached max steps.")
            request_heartbeat = False
        else:
            # Get request heartbeats and coerce to bool
            request_heartbeat = tool_args.pop("request_heartbeat", False)
            # Pre-emptively pop out inner_thoughts
            tool_args.pop(INNER_THOUGHTS_KWARG, "")

            # So this is necessary, because sometimes non-structured outputs makes mistakes
            if not isinstance(request_heartbeat, bool):
                if isinstance(request_heartbeat, str):
                    request_heartbeat = request_heartbeat.lower() == "true"
                else:
                    request_heartbeat = bool(request_heartbeat)

        tool_call_id = tool_call.id or f"call_{uuid.uuid4().hex[:8]}"

        debug_log(
            self.at_user_id,
            f"_handle_ai_response: tool={tool_call_name} is_final_step={is_final_step} request_heartbeat={request_heartbeat} tool_valid={tool_call_name in valid_tool_names}",
        )
        log_telemetry(
            self.logger,
            "_handle_ai_response execute tool start",
            tool_name=tool_call_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
            request_heartbeat=request_heartbeat,
        )
        if tool_call_name not in valid_tool_names:
            base_error_message = f"[ToolConstraintError] Cannot call {tool_call_name}, valid tools to call include: {valid_tool_names}."
            violated_rule_messages = tool_rules_solver.guess_rule_violation(tool_call_name)
            if violated_rule_messages:
                bullet_points = "\n".join(f"\t- {msg}" for msg in violated_rule_messages)
                base_error_message += f"\n** Hint: Possible rules that were violated:\n{bullet_points}"
            tool_execution_result = ToolExecutionResult(status="error", func_return=base_error_message)
            debug_log(self.at_user_id, f"_handle_ai_response: INVALID_TOOL tool={tool_call_name} violated_rules={violated_rule_messages}")
        else:
            async with AsyncTimer() as _t:
                tool_execution_result = await self._execute_tool(
                    tool_name=tool_call_name,
                    tool_args=tool_args,
                    agent_state=agent_state,
                    agent_step_span=agent_step_span,
                    step_id=step_id,
                )
            _log_step_timing(
                "tool_execution",
                _t.elapsed_ms,
                agent_id=agent_state.id,
                user_id=self.at_user_id,
                step_id=step_id,
                tool=tool_call_name,
                success=tool_execution_result.success_flag,
            )
            debug_log(
                self.at_user_id,
                f"_handle_ai_response: tool_executed tool={tool_call_name} success={tool_execution_result.success_flag} elapsed_ms={_t.elapsed_ms:.0f}",
            )
        log_telemetry(
            self.logger, "_handle_ai_response execute tool finish", tool_execution_result=tool_execution_result, tool_call_id=tool_call_id
        )

        if tool_call_name in ["conversation_search", "conversation_search_date", "archival_memory_search"]:
            # with certain functions we rely on the paging mechanism to handle overflow
            truncate = False
            debug_log(self.at_user_id, f"_handle_ai_response: MEMORY_SEARCH tool={tool_call_name} truncate=False (paging mode)")
        else:
            # but by default, we add a truncation safeguard to prevent bad functions from
            # overflow the agent context window
            truncate = True
            debug_log(self.at_user_id, f"_handle_ai_response: tool={tool_call_name} truncate=True")

        # get the function response limit
        target_tool = next((x for x in agent_state.tools if x.name == tool_call_name), None)
        return_char_limit = target_tool.return_char_limit
        function_response_string = validate_function_response(
            tool_execution_result.func_return, return_char_limit=return_char_limit, truncate=truncate
        )
        function_response = package_function_response(
            was_success=tool_execution_result.success_flag,
            response_string=function_response_string,
        )

        # 4. Register tool call with tool rule solver
        # Resolve whether or not to continue stepping
        continue_stepping = request_heartbeat
        tool_rules_solver.register_tool_call(tool_name=tool_call_name)
        if tool_rules_solver.is_terminal_tool(tool_name=tool_call_name):
            if continue_stepping:
                stop_reason = LettaStopReason(stop_reason=StopReasonType.tool_rule.value)
            continue_stepping = False
            debug_log(
                self.at_user_id,
                f"_handle_ai_response: TERMINAL_TOOL tool={tool_call_name} continue_stepping=False stop_reason={stop_reason}",
            )
        elif tool_rules_solver.has_children_tools(tool_name=tool_call_name):
            continue_stepping = True
            debug_log(self.at_user_id, f"_handle_ai_response: HAS_CHILDREN_TOOLS tool={tool_call_name} continue_stepping=True")
        elif tool_rules_solver.is_continue_tool(tool_name=tool_call_name):
            continue_stepping = True
            debug_log(self.at_user_id, f"_handle_ai_response: CONTINUE_TOOL tool={tool_call_name} continue_stepping=True")
        else:
            debug_log(
                self.at_user_id, f"_handle_ai_response: tool={tool_call_name} continue_stepping={continue_stepping} (request_heartbeat)"
            )

        # 5a. Persist Steps to DB
        # Following agent loop to persist this before messages
        # TODO (cliandy): determine what should match old loop w/provider_id, job_id
        # TODO (cliandy): UsageStatistics and LettaUsageStatistics are used in many places, but are not the same.
        async with AsyncTimer() as _t:
            logged_step = await self.step_manager.log_step_async(
                actor=self.actor,
                agent_id=agent_state.id,
                provider_name=agent_state.llm_config.model_endpoint_type,
                provider_category=agent_state.llm_config.provider_category or "base",
                model=agent_state.llm_config.model,
                model_endpoint=agent_state.llm_config.model_endpoint,
                context_window_limit=agent_state.llm_config.context_window,
                usage=usage,
                provider_id=None,
                job_id=None,
                step_id=step_id,
            )
        _log_step_timing("step_persist", _t.elapsed_ms, agent_id=agent_state.id, user_id=self.at_user_id, step_id=step_id)

        # 5b. Persist Messages to DB
        tool_call_messages = create_letta_messages_from_llm_response(
            agent_id=agent_state.id,
            model=agent_state.llm_config.model,
            function_name=tool_call_name,
            function_arguments=tool_args,
            tool_execution_result=tool_execution_result,
            tool_call_id=tool_call_id,
            function_call_success=tool_execution_result.success_flag,
            function_response=function_response_string,
            actor=self.actor,
            add_heartbeat_request_system_message=continue_stepping,
            reasoning_content=reasoning_content,
            pre_computed_assistant_message_id=pre_computed_assistant_message_id,
            step_id=logged_step.id if logged_step else None,  # TODO (cliandy): eventually move over other agent loops
        )

        # Do not persist system-role input messages (go-ai-chat sanity/date/reasoning rules).
        # They are re-sent fresh on every call, so historical copies accumulate in DB and
        # grow input tokens O(N) per turn. Filtering them here stops the accumulation
        # without any behaviour change — the LLM still receives them for the current call.
        # Validated for many days on userId=92744418 with no qualitative regression in
        # 20-30 turn conversations. Graduating universally: stale system-event copies in
        # history are noise that contradicts the fresh per-turn injections from go-ai-chat
        # (e.g., yesterday's "current time is 2:30 PM" conflicting with today's fresh stamp).
        _persistable_initial = [m for m in (initial_messages or []) if m.role != MessageRole.system]
        _all_messages_to_persist = _persistable_initial + tool_call_messages
        async with AsyncTimer() as _t:
            persisted_messages = await self.message_manager.create_many_messages_async(_all_messages_to_persist, actor=self.actor)
        _log_step_timing(
            "message_persist",
            _t.elapsed_ms,
            agent_id=agent_state.id,
            user_id=self.at_user_id,
            step_id=step_id,
            msg_count=len(_all_messages_to_persist),
        )
        self.last_function_response = function_response

        return persisted_messages, continue_stepping, stop_reason

    @trace_method
    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: JsonDict,
        agent_state: AgentState,
        agent_step_span: Optional["Span"] = None,
        step_id: str | None = None,
    ) -> "ToolExecutionResult":
        """
        Executes a tool and returns the ToolExecutionResult.
        """
        from letta.schemas.tool_execution_result import ToolExecutionResult

        # Special memory case
        target_tool = next((x for x in agent_state.tools if x.name == tool_name), None)
        if not target_tool:
            # TODO: fix this error message
            return ToolExecutionResult(
                func_return=f"Tool {tool_name} not found",
                status="error",
            )

        # TODO: This temp. Move this logic and code to executors

        if agent_step_span:
            start_time = get_utc_timestamp_ns()
            agent_step_span.add_event(name="tool_execution_started")

        sandbox_env_vars = {var.key: var.value for var in agent_state.tool_exec_environment_variables}
        tool_execution_manager = ToolExecutionManager(
            agent_state=agent_state,
            message_manager=self.message_manager,
            agent_manager=self.agent_manager,
            block_manager=self.block_manager,
            passage_manager=self.passage_manager,
            sandbox_env_vars=sandbox_env_vars,
            actor=self.actor,
            task_id=self._task_id,
        )
        # TODO: Integrate sandbox result
        _is_archival = tool_name in ("archival_memory_insert", "archival_memory_search", "archival_memory_delete")
        if _is_archival:
            debug_log(
                self.at_user_id,
                lambda _n=tool_name, _a=tool_args: f"_execute_tool: ARCHIVAL_MEMORY tool={_n} args={__import__('json').dumps(_a, default=str)}",
            )
        log_event(name=f"start_{tool_name}_execution", attributes=tool_args)
        tool_execution_result = await tool_execution_manager.execute_tool_async(
            function_name=tool_name,
            function_args=tool_args,
            tool=target_tool,
            step_id=step_id,
        )
        if _is_archival:
            debug_log(
                self.at_user_id,
                lambda _n=tool_name, _r=tool_execution_result: f"_execute_tool: ARCHIVAL_MEMORY_RESULT tool={_n} success={_r.success_flag} result={__import__('json').dumps(_r.func_return, default=str)}",
            )
        if agent_step_span:
            end_time = get_utc_timestamp_ns()
            agent_step_span.add_event(
                name="tool_execution_completed",
                attributes={
                    "tool_name": target_tool.name,
                    "duration_ms": ns_to_ms((end_time - start_time)),
                    "success": tool_execution_result.success_flag,
                    "tool_type": target_tool.tool_type,
                    "tool_id": target_tool.id,
                },
            )
        log_event(name=f"finish_{tool_name}_execution", attributes=tool_execution_result.model_dump())
        return tool_execution_result

    @trace_method
    def _load_last_function_response(self, in_context_messages: List[Message]):
        """Load the last function response from message history"""
        for msg in reversed(in_context_messages):
            if msg.role == MessageRole.tool and msg.content and len(msg.content) == 1 and isinstance(msg.content[0], TextContent):
                text_content = msg.content[0].text
                try:
                    response_json = json.loads(text_content)
                    if response_json.get("message"):
                        return response_json["message"]
                except (json.JSONDecodeError, KeyError):
                    raise ValueError(f"Invalid JSON format in message: {text_content}")
        return None
