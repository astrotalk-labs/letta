from typing import Any, Union

from pydantic import BaseModel, Field

from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.letta_message_content import TextContent
from letta.schemas.llm_config import LLMConfig


class CoreMemoryBlockSchema(BaseModel):
    created_at: str
    description: str | None
    is_template: bool
    label: str
    limit: int
    metadata_: dict | None = None
    template_name: str | None
    updated_at: str
    value: str


class MessageSchema(BaseModel):
    created_at: str
    group_id: str | None
    model: str | None
    name: str | None
    role: str
    content: list[TextContent]  # TODO: Expand to more in the future
    tool_call_id: str | None
    tool_calls: list[Any]
    tool_returns: list[Any]
    updated_at: str


class TagSchema(BaseModel):
    tag: str


class ToolEnvVarSchema(BaseModel):
    created_at: str
    description: str | None
    key: str
    updated_at: str
    value: str


# Tool rules


class BaseToolRuleSchema(BaseModel):
    tool_name: str
    type: str


class ChildToolRuleSchema(BaseToolRuleSchema):
    children: list[str]


class MaxCountPerStepToolRuleSchema(BaseToolRuleSchema):
    max_count_limit: int


class ConditionalToolRuleSchema(BaseToolRuleSchema):
    default_child: str | None
    child_output_mapping: dict[Any, str]
    require_output_mapping: bool


ToolRuleSchema = Union[BaseToolRuleSchema, ChildToolRuleSchema, MaxCountPerStepToolRuleSchema, ConditionalToolRuleSchema]


class ParameterProperties(BaseModel):
    type: str
    description: str | None = None


class ParametersSchema(BaseModel):
    type: str | None = "object"
    properties: dict[str, ParameterProperties]
    required: list[str] = Field(default_factory=list)


class ToolJSONSchema(BaseModel):
    name: str
    description: str
    parameters: ParametersSchema  # <— nested strong typing
    type: str | None = None  # top-level 'type' if it exists
    required: list[str] | None = Field(default_factory=list)


class ToolSchema(BaseModel):
    args_json_schema: Any | None
    created_at: str
    description: str
    json_schema: ToolJSONSchema
    name: str
    return_char_limit: int
    source_code: str | None
    source_type: str
    tags: list[str]
    tool_type: str
    updated_at: str
    metadata_: dict | None = None


class AgentSchema(BaseModel):
    agent_type: str
    core_memory: list[CoreMemoryBlockSchema]
    created_at: str
    description: str | None
    embedding_config: EmbeddingConfig
    llm_config: LLMConfig
    message_buffer_autoclear: bool
    in_context_message_indices: list[int]
    messages: list[MessageSchema]
    metadata_: dict | None = None
    multi_agent_group: Any | None
    name: str
    system: str
    tags: list[TagSchema]
    tool_exec_environment_variables: list[ToolEnvVarSchema]
    tool_rules: list[ToolRuleSchema]
    tools: list[ToolSchema]
    updated_at: str
    version: str
