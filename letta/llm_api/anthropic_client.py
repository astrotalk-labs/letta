import hashlib
import json
import logging
import re
from typing import Dict, List, Optional, Union

import anthropic
from anthropic import AsyncStream
from anthropic.types.beta import BetaMessage as AnthropicMessage
from anthropic.types.beta import BetaRawMessageStreamEvent
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages import BetaMessageBatch
from anthropic.types.beta.messages.batch_create_params import Request

from letta.errors import (
    ContextWindowExceededError,
    ErrorCode,
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMNotFoundError,
    LLMPermissionDeniedError,
    LLMRateLimitError,
    LLMServerError,
    LLMUnprocessableEntityError,
)
from letta.helpers.datetime_helpers import get_utc_time_int
from letta.llm_api.helpers import add_inner_thoughts_to_functions, unpack_all_inner_thoughts_from_kwargs
from letta.llm_api.llm_client_base import LLMClientBase
from letta.local_llm.constants import INNER_THOUGHTS_KWARG, INNER_THOUGHTS_KWARG_DESCRIPTION
from letta.log import get_logger
from letta.otel.tracing import trace_method
from letta.schemas.enums import ProviderCategory
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message as PydanticMessage
from letta.schemas.openai.chat_completion_request import Tool as OpenAITool
from letta.schemas.openai.chat_completion_response import ChatCompletionResponse, Choice, FunctionCall
from letta.schemas.openai.chat_completion_response import Message as ChoiceMessage
from letta.schemas.openai.chat_completion_response import ToolCall, UsageStatistics
from letta.services.provider_manager import ProviderManager
from letta.settings import model_settings

DUMMY_FIRST_USER_MESSAGE = "User initializing bootup sequence."

# Gate for cache-observability logs. Set to a single user_id who is always
# observed (force-included regardless of sampling). Empty string disables the
# always-on user. The v2 cache-optimization path is controlled by the
# V2_CACHE_ROLLOUT_* knobs below, not by this constant.
CACHE_OBS_USER_ID = "92744418"

# Broader sampling for cache-observability logs across all v2 traffic.
# Percentage of users whose (int(at_user_id) % 100) < CACHE_OBS_SAMPLE_PCT
# will have CACHE_OBS_REQ/USAGE/MISS_DIAG logs emitted in addition to the
# always-on CACHE_OBS_USER_ID. Set to 0 to disable broader sampling.
# At 1%, on ~22k calls/hour we expect ~220 sampled calls/hour = manageable
# log volume. We need this to determine the dominant cache-invalidation
# source in aggregate production traffic (not just one user).
CACHE_OBS_SAMPLE_PCT = 5

# Cold-miss threshold for CACHE_MISS_DIAG: emit diagnostic only when
# cache_read == 0 AND cache_creation > this many tokens. Filters out tiny
# incremental writes (just-the-new-message-tail cache writes) and focuses
# on the expensive full-prefix cold writes we want to understand.
CACHE_MISS_DIAG_MIN_CREATION_TOKENS = 3000


def _is_user_in_cache_obs_sample(at_user_id):
    # Returns True if this user's request should be logged for cache observability.
    # Combines (a) the always-on test user, and (b) a deterministic % sample of
    # broader traffic. False for missing/non-numeric ids so unknown traffic is
    # never accidentally logged.
    if not at_user_id:
        return False
    if CACHE_OBS_USER_ID and at_user_id == CACHE_OBS_USER_ID:
        return True
    pct = CACHE_OBS_SAMPLE_PCT
    if not pct or pct <= 0:
        return False
    if pct >= 100:
        return True
    try:
        return (int(at_user_id) % 100) < pct
    except (ValueError, TypeError):
        return False

# v2 cache-layout rollout: reverted to 0% after May 14 graduation regression.
# Per-order cost rose from $0.314 (v1 baseline May 11) to $0.366 (+17%) on
# May 14 when v2 went 100%. Anthropic API analysis: cache_creation share for
# Sonnet 4.5 jumped 53% -> 71% the same hour as graduation, and the new
# messages-tail breakpoint paired with mid-turn core_memory_append rebuilds
# causes cascade cache invalidation (cache_read collapses to 2114 on ~30%
# of calls, with 25-50k tokens of cache_creation per cold miss).
# Setting BUCKET_MAX=0 sends all production users back to v1. Test user
# CACHE_OBS_USER_ID stays on v2 (force-included below) so we can keep
# v1-vs-v2 comparable in obs logs while we ship the cascade fix on v1.
V2_CACHE_ROLLOUT_MOD = 10
V2_CACHE_ROLLOUT_BUCKET_MAX = 0


def _is_user_in_v2_cache_bucket(at_user_id):
    # Returns True if the user_id falls in the v2 rollout bucket. False for
    # missing/non-numeric ids so unknown traffic defaults to the v1 path (fail-closed).
    if not at_user_id:
        return False
    if CACHE_OBS_USER_ID and at_user_id == CACHE_OBS_USER_ID:
        return True
    if V2_CACHE_ROLLOUT_BUCKET_MAX <= 0:
        return False
    if V2_CACHE_ROLLOUT_BUCKET_MAX >= V2_CACHE_ROLLOUT_MOD:
        return True
    try:
        return (int(at_user_id) % V2_CACHE_ROLLOUT_MOD) < V2_CACHE_ROLLOUT_BUCKET_MAX
    except (ValueError, TypeError):
        return False


# --- Geography / business-based API key routing ---
_COHORT_TO_KEY_ENV: dict = {
    "AT_NATIVE":  "ABC2_AT_NATIVE_ANTHROPIC_KEY",
    "AT_FOREIGN": "ABC2_AT_FOREIGN_ANTHROPIC_KEY",
    "INDIAN_AT":  "ABC2_AT_INDIA_ANTHROPIC_KEY",
    "PANDITJI":   "ABC2_PANDITJI_ANTHROPIC_KEY",
    "LUMUS":      "ABC2_LUMUS_ANTHROPIC_KEY",
}

_COHORT_TO_BEDROCK_ARN: dict = {
    "AT_NATIVE":  "v9a7gf6hhoqm",
    "AT_FOREIGN": "ngef7rrnxwrh",
    "INDIAN_AT":  "rg76qwszdgvo",
    "PANDITJI":   "18c7gtd1mst5",
    "LUMUS":      "9gx7yplss61x",
}


def _build_bedrock_arn(profile_id: str) -> str:
    """Construct a full Bedrock ARN by taking the prefix from BEDROCK_INFERENCE_PROFILE_ARN
    and replacing its profile ID with the given one. Falls back to the raw ID if env var unset."""
    import os
    base = os.getenv("BEDROCK_INFERENCE_PROFILE_ARN", "")
    if "/" in base:
        prefix = base.rsplit("/", 1)[0]
        return f"{prefix}/{profile_id}"
    return profile_id


def _resolve_bedrock_arn_from_cohort(at_user_id: Optional[str], user_cohort: Optional[str]) -> Optional[str]:
    """Return the Bedrock inference profile ARN for the given cohort, or None to fall back
    to the existing BEDROCK_*_INFERENCE_PROFILE_ARN env vars."""
    if not user_cohort or user_cohort == "UNKNOWN":
        get_logger(__name__).warning("[GEO_KEY_BEDROCK] user=%s cohort=UNKNOWN/missing, using default ARN", at_user_id)
        return None
    profile_id = _COHORT_TO_BEDROCK_ARN.get(user_cohort)
    if not profile_id:
        get_logger(__name__).warning("[GEO_KEY_BEDROCK] user=%s unrecognised cohort=%s, using default ARN", at_user_id, user_cohort)
        return None
    arn = _build_bedrock_arn(profile_id)
    get_logger(__name__).info("[GEO_KEY_BEDROCK] user=%s cohort=%s → arn=%s", at_user_id, user_cohort, arn)
    return arn


def _resolve_key_from_cohort(at_user_id: Optional[str], user_cohort: Optional[str]) -> Optional[str]:
    """Return the Anthropic API key value for the given cohort, or None to use the default.
    UNKNOWN cohort and unrecognised values fall back to default."""
    import os
    if not user_cohort or user_cohort == "UNKNOWN":
        get_logger(__name__).warning("[GEO_KEY] user=%s cohort=UNKNOWN/missing, using default key", at_user_id)
        return None
    key_env = _COHORT_TO_KEY_ENV.get(user_cohort)
    if not key_env:
        get_logger(__name__).warning("[GEO_KEY] user=%s unrecognised cohort=%s, using default key", at_user_id, user_cohort)
        return None
    key_value = os.environ.get(key_env)
    if key_value:
        get_logger(__name__).info("[GEO_KEY] user=%s cohort=%s → env=%s", at_user_id, user_cohort, key_env)
    else:
        get_logger(__name__).warning("[GEO_KEY] user=%s cohort=%s env var %s not set, using default key", at_user_id, user_cohort, key_env)
    return key_value or None


logger = get_logger(__name__)


class AnthropicClient(LLMClientBase):

    @trace_method
    def request(self, request_data: dict, llm_config: LLMConfig) -> dict:
        client = self._get_anthropic_client(llm_config, async_client=False)
        response = client.beta.messages.create(**request_data, betas=["tools-2024-04-04", "prompt-caching-2024-07-31"])
        return response.model_dump()

    @trace_method
    async def request_async(self, request_data: dict, llm_config: LLMConfig, use_vertex_experiment: bool = False, use_bedrock_experiment: bool = False) -> dict:
        # Check if we should use Vertex AI based on model_endpoint_type
        # (This happens when use_vertex_experiment changes the endpoint type in _step)
        if llm_config.model_endpoint_type == "anthropic_vertex":
            print(f"DEBUG: [AnthropicClient] Using Vertex AI client (model_endpoint_type=anthropic_vertex)")
            print(f"DEBUG: [AnthropicClient] Request data model: {request_data.get('model')}")
            from anthropic import AsyncAnthropicVertex
            from letta.settings import model_settings
            import os

            # Create async Vertex client with proper configuration
            project_id = model_settings.google_cloud_project
            # Use 'global' region for newer Claude models
            region = os.getenv('GOOGLE_CLOUD_LOCATION') or os.getenv('ANTHROPIC_VERTEX_REGION') or 'global'

            print(f"DEBUG: [AnthropicClient] Creating AsyncAnthropicVertex with project={project_id}, region={region}")

            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT must be set for Vertex AI")

            # Override base_url to fix the 'global-aiplatform' issue
            # The SDK creates 'https://global-aiplatform.googleapis.com' but we need 'https://aiplatform.googleapis.com'
            base_url = "https://aiplatform.googleapis.com/v1/"

            client = AsyncAnthropicVertex(
                project_id=project_id,
                region=region,
                base_url=base_url,
            )

            print(f"DEBUG: [AnthropicClient] AsyncAnthropicVertex client created: {type(client)}")
            print(f"DEBUG: [AnthropicClient] Client base_url: {getattr(client, 'base_url', 'N/A')}")
            print(f"DEBUG: [AnthropicClient] Client _base_url: {getattr(client, '_base_url', 'N/A')}")

            # Vertex doesn't support beta features
            print(f"DEBUG: [AnthropicClient] Calling client.messages.create with model={request_data.get('model')}")
            response = await client.messages.create(**request_data)
            print(f"DEBUG: [AnthropicClient] Response received successfully")
        elif llm_config.model_endpoint_type == "anthropic_bedrock":
            print(f"DEBUG: [AnthropicClient] Using AWS Bedrock client via boto3 (model_endpoint_type=anthropic_bedrock)")
            print(f"DEBUG: [AnthropicClient] Request data model: {request_data.get('model')}")
            import os
            import json
            import asyncio
            import boto3
            from letta.settings import model_settings

            aws_region = os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION') or model_settings.aws_region or 'ap-south-1'
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID') or model_settings.aws_access_key
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY') or model_settings.aws_secret_access_key
            aws_session_token = os.getenv('AWS_SESSION_TOKEN')

            # Determine model ID: cohort-based ARN takes priority for gated users,
            # then fall back to model-name env vars, then raw model name.
            requested_model = (request_data.get('model') or llm_config.model or "").lower()
            logger.info("[GEO_KEY_BEDROCK] request_async at_user_id=%s user_cohort=%s", getattr(self, "at_user_id", None), getattr(self, "user_cohort", None))
            cohort_arn = _resolve_bedrock_arn_from_cohort(
                getattr(self, "at_user_id", None),
                getattr(self, "user_cohort", None),
            )
            if cohort_arn:
                bedrock_inference_profile = cohort_arn
            else:
                default_arn = os.getenv('BEDROCK_INFERENCE_PROFILE_ARN')
                if "haiku" in requested_model:
                    bedrock_inference_profile = os.getenv('BEDROCK_HAIKU_INFERENCE_PROFILE_ARN') or default_arn
                elif "sonnet-4-6" in requested_model or "sonnet-4.6" in requested_model:
                    bedrock_inference_profile = os.getenv('BEDROCK_SONNET_4_6_INFERENCE_PROFILE_ARN') or default_arn
                else:
                    bedrock_inference_profile = default_arn
            print(f"DEBUG: [AnthropicClient] Selected inference profile for model='{requested_model}': {bedrock_inference_profile}")
            model_id = bedrock_inference_profile if bedrock_inference_profile else request_data.get('model')

            print(f"DEBUG: [AnthropicClient] Creating boto3 bedrock-runtime client with region={aws_region}")
            print(f"DEBUG: [AnthropicClient] Using model_id={model_id}")

            client_kwargs = {
                "service_name": "bedrock-runtime",
                "region_name": aws_region,
            }
            if aws_access_key and aws_secret_key:
                client_kwargs["aws_access_key_id"] = aws_access_key
                client_kwargs["aws_secret_access_key"] = aws_secret_key
            if aws_session_token:
                client_kwargs["aws_session_token"] = aws_session_token

            bedrock_client = boto3.client(**client_kwargs)

            # Build the Bedrock request body in Anthropic Messages API format
            bedrock_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": request_data.get("max_tokens", 4096),
                "messages": request_data.get("messages", []),
            }
            if "system" in request_data:
                bedrock_body["system"] = request_data["system"]
            if "tools" in request_data:
                bedrock_body["tools"] = request_data["tools"]
            if "tool_choice" in request_data:
                bedrock_body["tool_choice"] = request_data["tool_choice"]
            if "temperature" in request_data:
                bedrock_body["temperature"] = request_data["temperature"]
            if "top_p" in request_data:
                bedrock_body["top_p"] = request_data["top_p"]
            if "top_k" in request_data:
                bedrock_body["top_k"] = request_data["top_k"]
            if "stop_sequences" in request_data:
                bedrock_body["stop_sequences"] = request_data["stop_sequences"]
            if "thinking" in request_data:
                bedrock_body["thinking"] = request_data["thinking"]

            print(f"DEBUG: [AnthropicClient] Calling bedrock invoke_model with modelId={model_id}")

            # Run synchronous boto3 call in a thread to avoid blocking the event loop
            def _invoke():
                resp = bedrock_client.invoke_model(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(bedrock_body),
                )
                return json.loads(resp["body"].read())

            result = await asyncio.to_thread(_invoke)
            print(f"DEBUG: [AnthropicClient] Bedrock response received successfully")

            # Convert Bedrock response to match Anthropic SDK response format
            # Bedrock returns the same format as Anthropic Messages API
            response_dict = {
                "id": result.get("id", "msg_bedrock"),
                "type": "message",
                "role": result.get("role", "assistant"),
                "content": result.get("content", []),
                "model": result.get("model", model_id),
                "stop_reason": result.get("stop_reason", "end_turn"),
                "stop_sequence": result.get("stop_sequence"),
                "usage": result.get("usage", {"input_tokens": 0, "output_tokens": 0}),
            }
            logger.info("This is the usage response from claude (bedrock) %s", response_dict.get("usage"))
            self._log_cache_observation_usage(
                response_dict.get("usage"),
                endpoint_type=llm_config.model_endpoint_type,
                model=request_data.get("model"),
            )
            return response_dict
        else:
            print(f"DEBUG: [AnthropicClient] Using standard Anthropic client (model_endpoint_type={llm_config.model_endpoint_type})")
            client = await self._get_anthropic_client_async(llm_config, async_client=True)
            response = await client.beta.messages.create(**request_data, betas=["tools-2024-04-04", "prompt-caching-2024-07-31"])
        logger.info("This is the usage response from claude %s", response.usage)
        self._log_cache_observation_usage(
            response.usage,
            endpoint_type=llm_config.model_endpoint_type,
            model=request_data.get("model"),
        )
        return response.model_dump()

    @trace_method
    async def stream_async(self, request_data: dict, llm_config: LLMConfig) -> AsyncStream[BetaRawMessageStreamEvent]:
        client = await self._get_anthropic_client_async(llm_config, async_client=True)
        request_data["stream"] = True
        return await client.beta.messages.create(**request_data, betas=["tools-2024-04-04", "prompt-caching-2024-07-31"])

    @trace_method
    async def send_llm_batch_request_async(
        self,
        agent_messages_mapping: Dict[str, List[PydanticMessage]],
        agent_tools_mapping: Dict[str, List[dict]],
        agent_llm_config_mapping: Dict[str, LLMConfig],
    ) -> BetaMessageBatch:
        """
        Sends a batch request to the Anthropic API using the provided agent messages and tools mappings.

        Args:
            agent_messages_mapping: A dict mapping agent_id to their list of PydanticMessages.
            agent_tools_mapping: A dict mapping agent_id to their list of tool dicts.
            agent_llm_config_mapping: A dict mapping agent_id to their LLM config

        Returns:
            BetaMessageBatch: The batch response from the Anthropic API.

        Raises:
            ValueError: If the sets of agent_ids in the two mappings do not match.
            Exception: Transformed errors from the underlying API call.
        """
        # Validate that both mappings use the same set of agent_ids.
        if set(agent_messages_mapping.keys()) != set(agent_tools_mapping.keys()):
            raise ValueError("Agent mappings for messages and tools must use the same agent_ids.")

        try:
            requests = {
                agent_id: self.build_request_data(
                    messages=agent_messages_mapping[agent_id],
                    llm_config=agent_llm_config_mapping[agent_id],
                    tools=agent_tools_mapping[agent_id],
                )
                for agent_id in agent_messages_mapping
            }

            client = await self._get_anthropic_client_async(list(agent_llm_config_mapping.values())[0], async_client=True)

            anthropic_requests = [
                Request(custom_id=agent_id, params=MessageCreateParamsNonStreaming(**params)) for agent_id, params in requests.items()
            ]

            batch_response = await client.beta.messages.batches.create(requests=anthropic_requests)

            return batch_response

        except Exception as e:
            # Enhance logging here if additional context is needed
            logger.error("Error during send_llm_batch_request_async.", exc_info=True)
            raise self.handle_llm_error(e)

    @trace_method
    def _get_anthropic_client(
        self, llm_config: LLMConfig, async_client: bool = False
    ) -> Union[anthropic.AsyncAnthropic, anthropic.Anthropic]:
        override_key = None
        if llm_config.provider_category == ProviderCategory.byok:
            override_key = ProviderManager().get_override_key(llm_config.provider_name, actor=self.actor)

        if not override_key:
            at_uid = getattr(self, "at_user_id", None)
            cohort = getattr(self, "user_cohort", None)
            logger.info("[GEO_KEY] _get_anthropic_client at_user_id=%s user_cohort=%s", at_uid, cohort)
            override_key = _resolve_key_from_cohort(at_uid, cohort)

        if async_client:
            return (
                anthropic.AsyncAnthropic(api_key=override_key, max_retries=model_settings.anthropic_max_retries)
                if override_key
                else anthropic.AsyncAnthropic(max_retries=model_settings.anthropic_max_retries)
            )
        return (
            anthropic.Anthropic(api_key=override_key, max_retries=model_settings.anthropic_max_retries)
            if override_key
            else anthropic.Anthropic(max_retries=model_settings.anthropic_max_retries)
        )

    @trace_method
    async def _get_anthropic_client_async(
        self, llm_config: LLMConfig, async_client: bool = False
    ) -> Union[anthropic.AsyncAnthropic, anthropic.Anthropic]:
        override_key = None
        if llm_config.provider_category == ProviderCategory.byok:
            override_key = await ProviderManager().get_override_key_async(llm_config.provider_name, actor=self.actor)

        if not override_key:
            at_uid = getattr(self, "at_user_id", None)
            cohort = getattr(self, "user_cohort", None)
            logger.info("[GEO_KEY] _get_anthropic_client_async at_user_id=%s user_cohort=%s", at_uid, cohort)
            override_key = _resolve_key_from_cohort(at_uid, cohort)

        if async_client:
            return (
                anthropic.AsyncAnthropic(api_key=override_key, max_retries=model_settings.anthropic_max_retries)
                if override_key
                else anthropic.AsyncAnthropic(max_retries=model_settings.anthropic_max_retries)
            )
        return (
            anthropic.Anthropic(api_key=override_key, max_retries=model_settings.anthropic_max_retries)
            if override_key
            else anthropic.Anthropic(max_retries=model_settings.anthropic_max_retries)
        )

    @trace_method
    def build_request_data(
        self,
        messages: List[PydanticMessage],
        llm_config: LLMConfig,
        tools: Optional[List[dict]] = None,
        force_tool_call: Optional[str] = None,
    ) -> dict:
        # TODO: This needs to get cleaned up. The logic here is pretty confusing.
        # TODO: I really want to get rid of prefixing, it's a recipe for disaster code maintenance wise
        prefix_fill = True
        if not self.use_tool_naming:
            raise NotImplementedError("Only tool calling supported on Anthropic API requests")

        if not llm_config.max_tokens:
            raise ValueError("Max  tokens must be set for anthropic")

        data = {
            "model": llm_config.model,
            "max_tokens": llm_config.max_tokens,
            "temperature": llm_config.temperature,
        }

        # Extended Thinking
        if llm_config.enable_reasoner:
            model_name = llm_config.model or ""
            endpoint_type = llm_config.model_endpoint_type or ""
            # Bedrock supports adaptive thinking on Sonnet/Opus 4.6 (no `effort` field accepted)
            is_bedrock = endpoint_type == "anthropic_bedrock"
            supports_adaptive_model = (
                "claude-sonnet-4-6" in model_name
                or "claude-opus-4-6" in model_name
            )
            if is_bedrock and supports_adaptive_model:
                # Bedrock does not support the `effort` field yet — adaptive only
                data["thinking"] = {"type": "adaptive"}
                logger.warning(
                    f"[THINKING] model={model_name} endpoint={endpoint_type} mode=adaptive (effort not supported on Bedrock)"
                )
            else:
                data["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": llm_config.max_reasoning_tokens,
                }
                logger.warning(
                    f"[THINKING] model={model_name} endpoint={endpoint_type} mode=enabled budget_tokens={llm_config.max_reasoning_tokens}"
                )
            # `temperature` may only be set to 1 when thinking is enabled. Please consult our documentation at https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#important-considerations-when-using-extended-thinking'
            data["temperature"] = 1.0

            # Silently disable prefix_fill for now
            prefix_fill = False

        # Tools
        # For an overview on tool choice:
        # https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview
        if not tools:
            # Special case for summarization path
            tools_for_request = None
            tool_choice = None
        elif llm_config.enable_reasoner:
            # NOTE: reasoning models currently do not allow for `any`
            tool_choice = {"type": "auto", "disable_parallel_tool_use": True}
            tools_for_request = [OpenAITool(function=f) for f in tools]
        elif force_tool_call is not None:
            tool_choice = {"type": "tool", "name": force_tool_call, "disable_parallel_tool_use": True}
            tools_for_request = [OpenAITool(function=f) for f in tools if f["name"] == force_tool_call]

            # need to have this setting to be able to put inner thoughts in kwargs
            if not llm_config.put_inner_thoughts_in_kwargs:
                logger.warning(
                    f"Force setting put_inner_thoughts_in_kwargs to True for Claude because there is a forced tool call: {force_tool_call}"
                )
                llm_config.put_inner_thoughts_in_kwargs = True
        else:
            if llm_config.put_inner_thoughts_in_kwargs:
                # tool_choice_type other than "auto" only plays nice if thinking goes inside the tool calls
                tool_choice = {"type": "any", "disable_parallel_tool_use": True}
            else:
                tool_choice = {"type": "auto", "disable_parallel_tool_use": True}
            tools_for_request = [OpenAITool(function=f) for f in tools] if tools is not None else None

        # Add tool choice
        if tool_choice:
            data["tool_choice"] = tool_choice

        # Add inner thoughts kwarg
        # TODO: Can probably make this more efficient
        if tools_for_request and len(tools_for_request) > 0 and llm_config.put_inner_thoughts_in_kwargs:
            tools_with_inner_thoughts = add_inner_thoughts_to_functions(
                functions=[t.function.model_dump() for t in tools_for_request],
                inner_thoughts_key=INNER_THOUGHTS_KWARG,
                inner_thoughts_description=INNER_THOUGHTS_KWARG_DESCRIPTION,
            )
            tools_for_request = [OpenAITool(function=f) for f in tools_with_inner_thoughts]

        # Sort tools by name before serialization to eliminate non-deterministic
        # ordering from the SQLAlchemy `tools` relationship in letta/orm/agent.py
        # (no `order_by` clause). Without this, tools come back in different orders
        # across calls and the request prefix hash changes — invalidating ALL cache
        # breakpoints downstream even though tool content is identical. PR #57
        # verified the fix mechanically for the gated test user (tools hash now
        # constant); this graduates to all traffic.
        if tools_for_request and len(tools_for_request) > 0:
            tools_for_request = sorted(tools_for_request, key=lambda t: t.function.name)

        if tools_for_request and len(tools_for_request) > 0:
            # TODO eventually enable parallel tool use
            data["tools"] = convert_tools_to_anthropic_format(tools_for_request)

        # Messages
        inner_thoughts_xml_tag = "thinking"

        # Move 'system' to the top level
        if messages[0].role != "system":
            raise RuntimeError(f"First message is not a system message, instead has role {messages[0].role}")
        system_content = messages[0].content if isinstance(messages[0].content, str) else messages[0].content[0].text

        # Capture agent_id from the message stream onto the client so downstream
        # observability hooks (CACHE_OBS_REQ/USAGE, CACHE_MISS_DIAG) can include
        # it in their payloads. The agent_id lets us group consecutive calls per
        # agent during log analysis and compute realized cache hit rate across
        # sequential turns. PydanticMessage carries .agent_id.
        try:
            self._cache_obs_agent_id = getattr(messages[0], "agent_id", None)
        except Exception:
            self._cache_obs_agent_id = None

        # Cache-optimization rollout: 50/50 split by user_id % 10. Users in the v2
        # bucket get the v2 splitter (keeps base instructions and tool_usage_rules
        # cached even when persona mutates), stabilized <memory_metadata>, and
        # cache_control on the last assistant message. The other 50% stay on v1.
        # Test user CACHE_OBS_USER_ID is always in the v2 bucket so observability
        # logs remain comparable to the PR #54 baseline.
        at_user_id_for_opt = getattr(self, "at_user_id", None)
        use_v2_caching = _is_user_in_v2_cache_bucket(at_user_id_for_opt)
        v2_split = self._split_system_message_v2_for_caching(system_content) if use_v2_caching else None
        if v2_split is not None:
            static_base, dynamic_persona, dynamic_user_memory, static_rules, dynamic_metadata = v2_split
            if dynamic_metadata:
                dynamic_metadata = self._stabilize_memory_metadata(dynamic_metadata)
            data["system"] = self._add_cache_control_to_system_message_v2(
                static_base, dynamic_persona, dynamic_user_memory, static_rules, dynamic_metadata
            )
        else:
            static_part_1, dynamic_part_1, static_part_2, dynamic_part_2 = self._split_system_message_for_caching(system_content)
            data["system"] = self._add_cache_control_to_system_message(static_part_1, dynamic_part_1, static_part_2, dynamic_part_2)
        data["messages"] = [
            m.to_anthropic_dict(
                inner_thoughts_xml_tag=inner_thoughts_xml_tag,
                put_inner_thoughts_in_kwargs=bool(llm_config.put_inner_thoughts_in_kwargs),
            )
            for m in messages[1:]
        ]


        # Ensure first message is user
        if not data["messages"] or data["messages"][0]["role"] != "user":
            data["messages"] = [{"role": "user", "content": DUMMY_FIRST_USER_MESSAGE}] + data["messages"]

        # Handle alternating messages
        data["messages"] = merge_tool_results_into_user_messages(data["messages"])

        # Gated: cache the conversation tail by attaching cache_control: ephemeral to the
        # last assistant message in the persisted history. With a stabilized system prefix
        # (v2 splitter + stabilized memory_metadata), the next call's prefix will match up
        # through this breakpoint and read everything from cache. Applied BEFORE prefix_fill
        # so the prefix-fill assistant marker (appended below) stays uncached and never sits
        # at the breakpoint position.
        if use_v2_caching:
            self._add_cache_control_to_last_assistant_message(data["messages"])

        # Prefix fill
        # https://docs.anthropic.com/en/api/messages#body-messages
        # NOTE: cannot prefill with tools for opus or Claude 4.6+:
        # Prefilling assistant messages is NOT supported on Claude 4.6 models (returns 400 error)
        model_name = data["model"]
        prefill_blocked = "opus" in model_name or "claude-sonnet-4-6" in model_name or "claude-opus-4-6" in model_name
        if prefix_fill and not llm_config.put_inner_thoughts_in_kwargs and not prefill_blocked:
            data["messages"].append(
                # Start the thinking process for the assistant
                {"role": "assistant", "content": f"<{inner_thoughts_xml_tag}>"},
            )

        self._log_cache_observation_request(data, num_input_messages=len(messages))

        return data

    def _split_system_message_for_caching(self, system_content: str) -> tuple:



        persona_end = system_content.find("</persona>")
        memory_blocks_end = system_content.find("</memory_blocks>")
        memory_metadata_start = system_content.find("<memory_metadata>")

        if persona_end != -1 and memory_blocks_end != -1:
            # Include closing tags in appropriate sections
            persona_end += len("</persona>")
            memory_blocks_end += len("</memory_blocks>")

            # Split into 4 parts
            static_part_1 = system_content[:persona_end].strip()        # Base + persona
            dynamic_part_1 = system_content[persona_end:memory_blocks_end].strip()  # human + conversation

            if memory_metadata_start != -1 and memory_metadata_start > memory_blocks_end:
                # Split the remaining content at memory_metadata
                static_part_2 = system_content[memory_blocks_end:memory_metadata_start].strip()  # tool_usage + files
                dynamic_part_2 = system_content[memory_metadata_start:].strip()  # memory_metadata
            else:
                # No memory_metadata found after memory_blocks
                static_part_2 = system_content[memory_blocks_end:].strip()  # tool_usage + files
                dynamic_part_2 = ""

            return static_part_1, dynamic_part_1, static_part_2, dynamic_part_2

        # Fallback - use your original 2-part split
        metadata_start = system_content.find("<memory_metadata>")
        if metadata_start != -1:
            static_part = system_content[:metadata_start].strip()
            dynamic_part = system_content[metadata_start:].strip()
            return static_part, dynamic_part, "", ""

        return system_content, "", "", ""

    def _add_cache_control_to_system_message(self, static_part_1, dynamic_part_1, static_part_2, dynamic_part_2):
        """Add cache control to 4-part system message structure"""
        system_parts = []

        # Add first static part with cache control (base instructions + persona)
        if static_part_1:
            system_parts.append({
                "type": "text",
                "text": static_part_1,
                "cache_control": {"type": "ephemeral"}
            })

        # Add first dynamic part without cache control (human + conversation_summary)
        if dynamic_part_1:
            system_parts.append({
                "type": "text",
                "text": dynamic_part_1
            })

        # Add second static part with cache control (tool_usage_rules + files)
        if static_part_2:
            system_parts.append({
                "type": "text",
                "text": static_part_2,
                "cache_control": {"type": "ephemeral"}
            })

        # Add second dynamic part without cache control (memory_metadata)
        if dynamic_part_2:
            system_parts.append({
                "type": "text",
                "text": dynamic_part_2
            })

        return system_parts

    def _split_system_message_v2_for_caching(self, system_content: str):
        # v2 splitter: 5-part layout that keeps base instructions and tool_usage_rules cached
        # even when the persona block is mutated by core_memory_append/replace.
        #
        # Layout vs v1:
        #   v1: [base+persona] cached | [human+summary] | [tool_usage+files] cached | [memory_metadata]
        #   v2: [base] cached | [persona] | [human+summary] | [tool_usage+files] cached | [memory_metadata]
        #
        # The trade-off: persona reads become uncached, but the base instructions (a larger,
        # truly stable chunk) stop being invalidated on every persona append. Net win because
        # cross-turn cache hit rate goes from near-zero to ~100% on the base prefix.
        #
        # Returns None when the markers needed for the 5-part split aren't present, so the
        # caller can fall back to v1.
        persona_start = system_content.find("<persona>")
        persona_end = system_content.find("</persona>")
        memory_blocks_end = system_content.find("</memory_blocks>")
        memory_metadata_start = system_content.find("<memory_metadata>")

        if persona_start == -1 or persona_end == -1 or memory_blocks_end == -1:
            return None
        if persona_start >= persona_end:
            return None

        persona_end += len("</persona>")
        memory_blocks_end += len("</memory_blocks>")

        static_base = system_content[:persona_start].strip()
        dynamic_persona = system_content[persona_start:persona_end].strip()
        dynamic_user_memory = system_content[persona_end:memory_blocks_end].strip()

        if memory_metadata_start != -1 and memory_metadata_start > memory_blocks_end:
            static_rules = system_content[memory_blocks_end:memory_metadata_start].strip()
            dynamic_metadata = system_content[memory_metadata_start:].strip()
        else:
            static_rules = system_content[memory_blocks_end:].strip()
            dynamic_metadata = ""

        return static_base, dynamic_persona, dynamic_user_memory, static_rules, dynamic_metadata

    def _add_cache_control_to_system_message_v2(self, static_base, dynamic_persona, dynamic_user_memory, static_rules, dynamic_metadata):
        # Build the 5-part system block list with cache_control: ephemeral on static_base and
        # static_rules only. Uses 2 of Anthropic's 4 available cache breakpoints, leaving room
        # for a third on the messages array.
        parts = []
        if static_base:
            parts.append({"type": "text", "text": static_base, "cache_control": {"type": "ephemeral"}})
        if dynamic_persona:
            parts.append({"type": "text", "text": dynamic_persona})
        if dynamic_user_memory:
            parts.append({"type": "text", "text": dynamic_user_memory})
        if static_rules:
            parts.append({"type": "text", "text": static_rules, "cache_control": {"type": "ephemeral"}})
        if dynamic_metadata:
            parts.append({"type": "text", "text": dynamic_metadata})
        return parts

    def _stabilize_memory_metadata(self, metadata_content: str) -> str:
        # Replace per-second timestamp + recall/archival counts with hourly-stable text so
        # this trailing block stops invalidating cache lookups on every turn. The agent still
        # gets a coarse time reference and a reminder that recall tools exist; it never acts
        # on exact recall/archival counts.
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc)
        hour_stamp = now.strftime("%Y-%m-%d %H UTC")
        return (
            "<memory_metadata>\n"
            f"- Current hour: {hour_stamp}\n"
            "- Recall and archival memory tools are available "
            "(conversation_search, archival_memory_search).\n"
            "</memory_metadata>"
        )

    def _add_cache_control_to_last_assistant_message(self, messages_list) -> None:
        # Walk the messages array backwards, find the last assistant message in the persisted
        # history, and attach cache_control: ephemeral to its last content block. This caches
        # the entire conversation tail through that point. Next call has the same prefix +
        # a new user turn, so it reads everything up to and including this message from cache.
        if not messages_list:
            return
        last_assistant_idx = None
        for i in range(len(messages_list) - 1, -1, -1):
            msg = messages_list[i]
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                last_assistant_idx = i
                break
        if last_assistant_idx is None:
            return
        msg = messages_list[last_assistant_idx]
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        elif isinstance(content, list) and content:
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = {"type": "ephemeral"}

    def _log_cache_observation_request(self, data: dict, num_input_messages: int) -> None:
        # Emits a structured log line describing the request prefix structure
        # so we can correlate it with the cache_read/cache_creation token counts
        # in the response. Gated by `_is_user_in_cache_obs_sample` which combines
        # an always-on test user with a small-percentage sample of broader traffic.
        # `at_user_id` is set by LLMClientBase.__init__; use getattr so a code
        # path that instantiates without going through the base init still no-ops.
        at_user_id = getattr(self, "at_user_id", None)
        if not _is_user_in_cache_obs_sample(at_user_id):
            return
        try:
            def _hash_short(text: str) -> str:
                return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:8]

            system_blocks = data.get("system", []) or []
            sys_summary = []
            dyn_tail_preview = ""
            for i, blk in enumerate(system_blocks):
                if isinstance(blk, dict):
                    text = blk.get("text", "") or ""
                    cached = bool(blk.get("cache_control"))
                else:
                    text = str(blk)
                    cached = False
                sys_summary.append({
                    "idx": i,
                    "chars": len(text),
                    "approx_tokens": len(text) // 4,
                    "cache_control": cached,
                    "prefix": text[:80].replace("\n", " "),
                    "hash": _hash_short(text),
                })
            # Log the trailing block in full when uncached - this is the prime suspect
            # for invalidating downstream cache lookups (memory_metadata).
            if system_blocks:
                last = system_blocks[-1]
                if isinstance(last, dict) and not last.get("cache_control"):
                    dyn_tail_preview = (last.get("text", "") or "")[:2000]

            tools = data.get("tools", []) or []
            tools_json = json.dumps(tools, sort_keys=True, default=str) if tools else ""
            tools_info = {
                "count": len(tools),
                "approx_tokens": len(tools_json) // 4,
                "hash": _hash_short(tools_json) if tools_json else "",
            }

            messages = data.get("messages", []) or []
            msg_summary = []
            role_counts: Dict[str, int] = {}
            total_msg_chars = 0
            messages_tail_cache_idx = None
            messages_tail_prefix_chars = 0
            running_prefix_chars = 0
            for i, msg in enumerate(messages):
                role = msg.get("role", "?")
                role_counts[role] = role_counts.get(role, 0) + 1
                content = msg.get("content", "")
                # Detect cache_control on this message (the v2 tail breakpoint
                # is placed on the last assistant message). When present, capture
                # the position and the running prefix size up to and including
                # this message so we can compute the cached portion size and
                # compare against cache_read_input_tokens from the response.
                has_cache_ctrl = False
                if isinstance(content, list):
                    ctext = json.dumps(content, sort_keys=True, default=str)
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("cache_control"):
                            has_cache_ctrl = True
                            break
                else:
                    ctext = str(content)
                total_msg_chars += len(ctext)
                running_prefix_chars += len(ctext)
                if has_cache_ctrl:
                    messages_tail_cache_idx = i
                    messages_tail_prefix_chars = running_prefix_chars
                preview = ctext[:80].replace("\n", " ")
                msg_summary.append({
                    "idx": i,
                    "role": role,
                    "chars": len(ctext),
                    "hash": _hash_short(ctext),
                    "cache_control": has_cache_ctrl,
                    "prefix": preview,
                })

            # Sum the chars/tokens of cached system blocks. Together with the
            # messages-tail cached prefix, this is the theoretical maximum
            # cache_read tokens for this call. Comparing against the response's
            # cache_read_input_tokens tells us how much of the available cache
            # actually hit.
            cached_system_chars = sum(b["chars"] for b in sys_summary if b.get("cache_control"))
            v2_active = _is_user_in_v2_cache_bucket(at_user_id)

            payload = {
                "at_user_id": at_user_id,
                "agent_id": getattr(self, "_cache_obs_agent_id", None),
                "model": data.get("model"),
                "v2_active": v2_active,
                "input_message_count": num_input_messages,
                "system_block_count": len(system_blocks),
                "system_cached_chars": cached_system_chars,
                "system_cached_approx_tokens": cached_system_chars // 4,
                "system_blocks": sys_summary,
                "dynamic_tail_preview": dyn_tail_preview,
                "tools": tools_info,
                "messages_count": len(messages),
                "messages_role_counts": role_counts,
                "messages_total_chars": total_msg_chars,
                "messages_total_approx_tokens": total_msg_chars // 4,
                "messages_tail_cache_idx": messages_tail_cache_idx,
                "messages_tail_prefix_chars": messages_tail_prefix_chars,
                "messages_tail_prefix_approx_tokens": messages_tail_prefix_chars // 4,
                "messages": msg_summary,
            }
            logger.info("[CACHE_OBS_REQ] %s", json.dumps(payload, default=str))
        except Exception as e:
            logger.warning("[CACHE_OBS_REQ] logging failed: %s", e)

    def _log_cache_observation_usage(self, usage, endpoint_type: Optional[str], model: Optional[str]) -> None:
        # Emits Anthropic usage with cache_read / cache_creation tokens so we can
        # measure the effect of caching changes. Gated to the same sample as
        # _log_cache_observation_request. Also emits CACHE_MISS_DIAG on cold misses
        # (cache_read == 0 AND cache_creation > threshold) so we can identify what
        # the request looked like when caching failed to engage.
        at_user_id = getattr(self, "at_user_id", None)
        if not _is_user_in_cache_obs_sample(at_user_id):
            return
        try:
            if hasattr(usage, "model_dump"):
                usage_dict = usage.model_dump()
            elif isinstance(usage, dict):
                usage_dict = usage
            else:
                usage_dict = {"raw": str(usage)}
            cache_create = usage_dict.get("cache_creation_input_tokens") or 0
            cache_read = usage_dict.get("cache_read_input_tokens") or 0
            payload = {
                "at_user_id": at_user_id,
                "agent_id": getattr(self, "_cache_obs_agent_id", None),
                "model": model,
                "endpoint_type": endpoint_type,
                "input_tokens": usage_dict.get("input_tokens"),
                "output_tokens": usage_dict.get("output_tokens"),
                "cache_creation_input_tokens": cache_create,
                "cache_read_input_tokens": cache_read,
            }
            logger.info("[CACHE_OBS_USAGE] %s", json.dumps(payload, default=str))

            # Cold-miss diagnostic: when cache_read is zero but cache_creation is
            # large, the cache was completely missed and we wrote a fresh full
            # prefix. Tag this so we can grep for the request shapes that cause it.
            if cache_read == 0 and cache_create >= CACHE_MISS_DIAG_MIN_CREATION_TOKENS:
                diag = {
                    "at_user_id": at_user_id,
                    "model": model,
                    "endpoint_type": endpoint_type,
                    "cache_creation_input_tokens": cache_create,
                    "input_tokens": usage_dict.get("input_tokens"),
                    "agent_id": getattr(self, "_cache_obs_agent_id", None),
                }
                logger.info("[CACHE_MISS_DIAG] %s", json.dumps(diag, default=str))
        except Exception as e:
            logger.warning("[CACHE_OBS_USAGE] logging failed: %s", e)

    async def count_tokens(self, messages: List[dict] = None, model: str = None, tools: List[OpenAITool] = None) -> int:
        logging.getLogger("httpx").setLevel(logging.WARNING)

        client = anthropic.AsyncAnthropic()
        if messages and len(messages) == 0:
            messages = None
        if tools and len(tools) > 0:
            anthropic_tools = convert_tools_to_anthropic_format(tools)
        else:
            anthropic_tools = None

        try:
            result = await client.beta.messages.count_tokens(
                model=model or "claude-3-7-sonnet-20250219",
                messages=messages or [{"role": "user", "content": "hi"}],
                tools=anthropic_tools or [],
            )
        except:
            raise

        token_count = result.input_tokens
        if messages is None:
            token_count -= 8
        return token_count

    @trace_method
    def handle_llm_error(self, e: Exception) -> Exception:
        if isinstance(e, anthropic.APIConnectionError):
            logger.warning(f"[Anthropic] API connection error: {e.__cause__}")
            return LLMConnectionError(
                message=f"Failed to connect to Anthropic: {str(e)}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                details={"cause": str(e.__cause__) if e.__cause__ else None},
            )

        if isinstance(e, anthropic.RateLimitError):
            logger.warning("[Anthropic] Rate limited (429). Consider backoff.")
            return LLMRateLimitError(
                message=f"Rate limited by Anthropic: {str(e)}",
                code=ErrorCode.RATE_LIMIT_EXCEEDED,
            )

        if isinstance(e, anthropic.BadRequestError):
            logger.warning(f"[Anthropic] Bad request: {str(e)}")
            if "prompt is too long" in str(e).lower():
                # If the context window is too large, we expect to receive:
                # 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'prompt is too long: 200758 tokens > 200000 maximum'}}
                return ContextWindowExceededError(
                    message=f"Bad request to Anthropic (context window exceeded): {str(e)}",
                )
            else:
                return LLMBadRequestError(
                    message=f"Bad request to Anthropic: {str(e)}",
                    code=ErrorCode.INTERNAL_SERVER_ERROR,
                )

        if isinstance(e, anthropic.AuthenticationError):
            logger.warning(f"[Anthropic] Authentication error: {str(e)}")
            return LLMAuthenticationError(
                message=f"Authentication failed with Anthropic: {str(e)}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
            )

        if isinstance(e, anthropic.PermissionDeniedError):
            logger.warning(f"[Anthropic] Permission denied: {str(e)}")
            return LLMPermissionDeniedError(
                message=f"Permission denied by Anthropic: {str(e)}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
            )

        if isinstance(e, anthropic.NotFoundError):
            logger.warning(f"[Anthropic] Resource not found: {str(e)}")
            return LLMNotFoundError(
                message=f"Resource not found in Anthropic: {str(e)}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
            )

        if isinstance(e, anthropic.UnprocessableEntityError):
            logger.warning(f"[Anthropic] Unprocessable entity: {str(e)}")
            return LLMUnprocessableEntityError(
                message=f"Invalid request content for Anthropic: {str(e)}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
            )

        if isinstance(e, anthropic.APIStatusError):
            logger.warning(f"[Anthropic] API status error: {str(e)}")
            return LLMServerError(
                message=f"Anthropic API error: {str(e)}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                details={
                    "status_code": e.status_code if hasattr(e, "status_code") else None,
                    "response": str(e.response) if hasattr(e, "response") else None,
                },
            )

        return super().handle_llm_error(e)

    # TODO: Input messages doesn't get used here
    # TODO: Clean up this interface
    @trace_method
    def convert_response_to_chat_completion(
        self,
        response_data: dict,
        input_messages: List[PydanticMessage],
        llm_config: LLMConfig,
    ) -> ChatCompletionResponse:
        """
        Example response from Claude 3:
        response.json = {
            'id': 'msg_01W1xg9hdRzbeN2CfZM7zD2w',
            'type': 'message',
            'role': 'assistant',
            'content': [
                {
                    'type': 'text',
                    'text': "<thinking>Analyzing user login event. This is Chad's first
        interaction with me. I will adjust my personality and rapport accordingly.</thinking>"
                },
                {
                    'type':
                    'tool_use',
                    'id': 'toolu_01Ka4AuCmfvxiidnBZuNfP1u',
                    'name': 'core_memory_append',
                    'input': {
                        'name': 'human',
                        'content': 'Chad is logging in for the first time. I will aim to build a warm
        and welcoming rapport.',
                        'request_heartbeat': True
                    }
                }
            ],
            'model': 'claude-3-haiku-20240307',
            'stop_reason': 'tool_use',
            'stop_sequence': None,
            'usage': {
                'input_tokens': 3305,
                'output_tokens': 141
            }
        }
        """
        response = AnthropicMessage(**response_data)
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        finish_reason = remap_finish_reason(str(response.stop_reason))

        content = None
        reasoning_content = None
        reasoning_content_signature = None
        redacted_reasoning_content = None
        tool_calls = None

        if len(response.content) > 0:
            for content_part in response.content:
                if content_part.type == "text":
                    content = strip_xml_tags(string=content_part.text, tag="thinking")
                if content_part.type == "tool_use":
                    # hack for incorrect tool format
                    tool_input = json.loads(json.dumps(content_part.input))
                    if "id" in tool_input and tool_input["id"].startswith("toolu_") and "function" in tool_input:
                        arguments = json.dumps(tool_input["function"]["arguments"], indent=2)
                        try:
                            args_json = json.loads(arguments)
                            if not isinstance(args_json, dict):
                                raise ValueError("Expected parseable json object for arguments")
                        except:
                            arguments = str(tool_input["function"]["arguments"])
                    else:
                        arguments = json.dumps(tool_input, indent=2)
                    tool_calls = [
                        ToolCall(
                            id=content_part.id,
                            type="function",
                            function=FunctionCall(
                                name=content_part.name,
                                arguments=arguments,
                            ),
                        )
                    ]
                if content_part.type == "thinking":
                    reasoning_content = content_part.thinking
                    reasoning_content_signature = content_part.signature
                if content_part.type == "redacted_thinking":
                    redacted_reasoning_content = content_part.data

        else:
            raise RuntimeError("Unexpected empty content in response")

        assert response.role == "assistant"
        choice = Choice(
            index=0,
            finish_reason=finish_reason,
            message=ChoiceMessage(
                role=response.role,
                content=content,
                reasoning_content=reasoning_content,
                reasoning_content_signature=reasoning_content_signature,
                redacted_reasoning_content=redacted_reasoning_content,
                tool_calls=tool_calls,
            ),
        )

        chat_completion_response = ChatCompletionResponse(
            id=response.id,
            choices=[choice],
            created=get_utc_time_int(),
            model=response.model,
            usage=UsageStatistics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        if llm_config.put_inner_thoughts_in_kwargs:
            chat_completion_response = unpack_all_inner_thoughts_from_kwargs(
                response=chat_completion_response, inner_thoughts_key=INNER_THOUGHTS_KWARG
            )

        return chat_completion_response


def convert_tools_to_anthropic_format(tools: List[OpenAITool]) -> List[dict]:
    """See: https://docs.anthropic.com/claude/docs/tool-use

    OpenAI style:
      "tools": [{
        "type": "function",
        "function": {
            "name": "find_movies",
            "description": "find ....",
            "parameters": {
              "type": "object",
              "properties": {
                 PARAM: {
                   "type": PARAM_TYPE,  # eg "string"
                   "description": PARAM_DESCRIPTION,
                 },
                 ...
              },
              "required": List[str],
            }
        }
      }
      ]

    Anthropic style:
      "tools": [{
        "name": "find_movies",
        "description": "find ....",
        "input_schema": {
          "type": "object",
          "properties": {
             PARAM: {
               "type": PARAM_TYPE,  # eg "string"
               "description": PARAM_DESCRIPTION,
             },
             ...
          },
          "required": List[str],
        }
      }
      ]

      Two small differences:
        - 1 level less of nesting
        - "parameters" -> "input_schema"
    """
    formatted_tools = []
    for tool in tools:
        formatted_tool = {
            "name": tool.function.name,
            "description": tool.function.description if tool.function.description else "",
            "input_schema": tool.function.parameters or {"type": "object", "properties": {}, "required": []},
        }
        formatted_tools.append(formatted_tool)

    return formatted_tools


def merge_tool_results_into_user_messages(messages: List[dict]):
    """Anthropic API doesn't allow role 'tool'->'user' sequences

    Example HTTP error:
    messages: roles must alternate between "user" and "assistant", but found multiple "user" roles in a row

    From: https://docs.anthropic.com/claude/docs/tool-use
    You may be familiar with other APIs that return tool use as separate from the model's primary output,
    or which use a special-purpose tool or function message role.
    In contrast, Anthropic's models and API are built around alternating user and assistant messages,
    where each message is an array of rich content blocks: text, image, tool_use, and tool_result.
    """

    # TODO walk through the messages list
    # When a dict (dict_A) with 'role' == 'user' is followed by a dict with 'role' == 'user' (dict B), do the following
    # dict_A["content"] = dict_A["content"] + dict_B["content"]

    # The result should be a new merged_messages list that doesn't have any back-to-back dicts with 'role' == 'user'
    merged_messages = []
    if not messages:
        return merged_messages

    # Start with the first message in the list
    current_message = messages[0]

    for next_message in messages[1:]:
        if current_message["role"] == "user" and next_message["role"] == "user":
            # Merge contents of the next user message into current one
            current_content = (
                current_message["content"]
                if isinstance(current_message["content"], list)
                else [{"type": "text", "text": current_message["content"]}]
            )
            next_content = (
                next_message["content"]
                if isinstance(next_message["content"], list)
                else [{"type": "text", "text": next_message["content"]}]
            )
            merged_content = current_content + next_content
            current_message["content"] = merged_content
        else:
            # Append the current message to result as it's complete
            merged_messages.append(current_message)
            # Move on to the next message
            current_message = next_message

    # Append the last processed message to the result
    merged_messages.append(current_message)

    return merged_messages


def remap_finish_reason(stop_reason: str) -> str:
    """Remap Anthropic's 'stop_reason' to OpenAI 'finish_reason'

    OpenAI: 'stop', 'length', 'function_call', 'content_filter', null
    see: https://platform.openai.com/docs/guides/text-generation/chat-completions-api

    From: https://docs.anthropic.com/claude/reference/migrating-from-text-completions-to-messages#stop-reason

    Messages have a stop_reason of one of the following values:
        "end_turn": The conversational turn ended naturally.
        "stop_sequence": One of your specified custom stop sequences was generated.
        "max_tokens": (unchanged)

    """
    if stop_reason == "end_turn":
        return "stop"
    elif stop_reason == "stop_sequence":
        return "stop"
    elif stop_reason == "max_tokens":
        return "length"
    elif stop_reason == "tool_use":
        return "function_call"
    else:
        raise ValueError(f"Unexpected stop_reason: {stop_reason}")


def strip_xml_tags(string: str, tag: Optional[str]) -> str:
    if tag is None:
        return string
    # Construct the regular expression pattern to find the start and end tags
    tag_pattern = f"<{tag}.*?>|</{tag}>"
    # Use the regular expression to replace the tags with an empty string
    return re.sub(tag_pattern, "", string)


def strip_xml_tags_streaming(string: str, tag: Optional[str]) -> str:
    if tag is None:
        return string

    # Handle common partial tag cases
    parts_to_remove = [
        "<",  # Leftover start bracket
        f"<{tag}",  # Opening tag start
        f"</{tag}",  # Closing tag start
        f"/{tag}>",  # Closing tag end
        f"{tag}>",  # Opening tag end
        f"/{tag}",  # Partial closing tag without >
        ">",  # Leftover end bracket
    ]

    result = string
    for part in parts_to_remove:
        result = result.replace(part, "")

    return result
