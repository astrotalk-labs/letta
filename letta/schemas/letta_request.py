from typing import Any, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, HttpUrl, field_validator

from letta.constants import DEFAULT_MAX_STEPS, DEFAULT_MESSAGE_TOOL, DEFAULT_MESSAGE_TOOL_KWARG
from letta.schemas.letta_message import MessageType
from letta.schemas.message import MessageCreate


class LettaRequest(BaseModel):
    # Allow callers to send either the camelCase JSON key or the snake_case Python
    # name for fields that declare a validation_alias (e.g. latencyOptimisationFlow).
    model_config = ConfigDict(populate_by_name=True)

    messages: List[MessageCreate] = Field(..., description="The messages to be sent to the agent.")
    max_steps: int = Field(
        default=DEFAULT_MAX_STEPS,
        description="Maximum number of steps the agent should take to process the request.",
    )
    use_assistant_message: bool = Field(
        default=True,
        description="Whether the server should parse specific tool call arguments (default `send_message`) as `AssistantMessage` objects.",
    )
    assistant_message_tool_name: str = Field(
        default=DEFAULT_MESSAGE_TOOL,
        description="The name of the designated message tool.",
    )
    assistant_message_tool_kwarg: str = Field(
        default=DEFAULT_MESSAGE_TOOL_KWARG,
        description="The name of the message argument in the designated message tool.",
    )

    # filter to only return specific message types
    include_return_message_types: Optional[List[MessageType]] = Field(
        default=None, description="Only return specified message types in the response. If `None` (default) returns all messages."
    )

    use_vertex_experiment: bool = Field(
        default=False,
        description="Flag to dynamically switch between Anthropic direct API and Vertex AI based on experiment configuration.",
    )

    use_bedrock_experiment: bool = Field(
        default=False,
        description="Flag to dynamically switch between Anthropic direct API and AWS Bedrock based on experiment configuration.",
    )

    model_override: Optional[str] = Field(
        default=None,
        description="Optional model name to use for this request only, overriding the agent's configured llm_config.model. Useful for runtime A/B testing (e.g. shifting an agent from Sonnet 4.5 to Sonnet 4.6 without modifying the stored agent config).",
    )

    user_cohort: Optional[str] = Field(
        default=None,
        description="User cohort for API key routing. One of: AT_NATIVE, AT_FOREIGN, INDIAN_AT, PANDITJI, LUMUS, UNKNOWN. Populated by the upstream service from Redis USER_STATIC_DATA_{userId}.",
    )

    business_id: Optional[int] = Field(
        default=None,
        description="Raw business ID from the order (1=AstroTalk, 10=Panditji, 12=Lumus). 0 when unavailable.",
    )

    chat_order_id: Optional[int] = Field(
        default=None,
        description="Chat order ID from the upstream order system. Used for debug log correlation.",
    )

    consultant_id: int | None = Field(
        default=None,
        description="Consultant identifier from the upstream order. When provided alongside llm_provider='bedrock', pins the call to the dedicated consultant Bedrock inference profile.",
    )

    llm_provider: Optional[str] = Field(
        default=None,
        description="Override the LLM provider for this request. Use 'google' to route to Google AI (Gemini) regardless of the agent's configured endpoint.",
    )

    thinking: Optional[dict] = Field(
        default=None,
        description="Extended thinking config forwarded to the Claude API (e.g. {'type': 'adaptive'}). When set, temperature is forced to 1 and max_tokens is raised to at least 8000.",
    )

    thinking_config: Optional[dict] = Field(
        default=None,
        description="Thinking config forwarded to Google AI / Gemini (e.g. {'level': 'low'}). Maps level ('low'|'medium'|'high') to a thinking_budget token count.",
    )

    output_config: Optional[dict] = Field(
        default=None,
        description="Output config forwarded to the Claude API (e.g. {'effort': 'high'}).",
    )

    task_id: Optional[str] = Field(
        default=None,
        description="Optional task ID for step-wise latency logging. When provided, timing for each phase (context prep, LLM call, tool execution, context rebuild) is emitted as warning logs to help identify bottlenecks.",
    )

    latencyOptimisationFlow: bool = Field(
        default=False,
        validation_alias=AliasChoices("latencyOptimisationFlow", "latency_optimisation_flow"),
        description=(
            "When True, latency-optimised execution is enabled: memory-tool follow-up steps "
            "(archival_memory_search, recall_memory_search, core_memory_append, "
            "core_memory_replace, archival_memory_insert) are routed through the configured "
            "Haiku model instead of the agent's default model. Step 0 and non-memory steps "
            "still use the full model so quality of reasoning and the final send_message "
            "response are unaffected. Works with Anthropic direct, Vertex AI, and AWS Bedrock. "
            "Accepts both camelCase (`latencyOptimisationFlow`) and snake_case "
            "(`latency_optimisation_flow`) JSON keys."
        ),
    )

    @field_validator("task_id", mode="before")
    @classmethod
    def coerce_task_id_to_str(cls, v: Any) -> Optional[str]:
        """Accept integers or any scalar as task_id and coerce to string."""
        if v is None:
            return None
        return str(v)


class LettaStreamingRequest(LettaRequest):
    stream_tokens: bool = Field(
        default=False,
        description="Flag to determine if individual tokens should be streamed. Set to True for token streaming (requires stream_steps = True).",
    )


class LettaBatchRequest(LettaRequest):
    agent_id: str = Field(..., description="The ID of the agent to send this batch request for")


class CreateBatch(BaseModel):
    requests: List[LettaBatchRequest] = Field(..., description="List of requests to be processed in batch.")
    callback_url: Optional[HttpUrl] = Field(
        None,
        description="Optional URL to call via POST when the batch completes. The callback payload will be a JSON object with the following fields: "
        "{'job_id': string, 'status': string, 'completed_at': string}. "
        "Where 'job_id' is the unique batch job identifier, "
        "'status' is the final batch status (e.g., 'completed', 'failed'), and "
        "'completed_at' is an ISO 8601 timestamp indicating when the batch job completed.",
    )
