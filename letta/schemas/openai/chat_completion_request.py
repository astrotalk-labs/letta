from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator


class SystemMessage(BaseModel):
    content: str
    role: str = "system"
    name: str | None = None


class UserMessage(BaseModel):
    content: str | list[str] | list[dict]
    role: str = "user"
    name: str | None = None


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class AssistantMessage(BaseModel):
    content: str | None = None
    role: str = "assistant"
    name: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolMessage(BaseModel):
    content: str
    role: str = "tool"
    tool_call_id: str


ChatMessage = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


# TODO: this might not be necessary with the validator
def cast_message_to_subtype(m_dict: dict) -> ChatMessage:
    """Cast a dictionary to one of the individual message types"""
    role = m_dict.get("role")
    if role == "system" or role == "developer":
        return SystemMessage(**m_dict)
    elif role == "user":
        return UserMessage(**m_dict)
    elif role == "assistant":
        return AssistantMessage(**m_dict)
    elif role == "tool":
        return ToolMessage(**m_dict)
    else:
        raise ValueError(f"Unknown message role: {role}")


class ResponseFormat(BaseModel):
    type: str = Field(default="text", pattern="^(text|json_object)$")


## tool_choice ##
class FunctionCall(BaseModel):
    name: str


class ToolFunctionChoice(BaseModel):
    # The type of the tool. Currently, only function is supported
    type: Literal["function"] = "function"
    # type: str = Field(default="function", const=True)
    function: FunctionCall


class AnthropicToolChoiceTool(BaseModel):
    type: str = "tool"
    name: str
    disable_parallel_tool_use: bool | None = False


class AnthropicToolChoiceAny(BaseModel):
    type: str = "any"
    disable_parallel_tool_use: bool | None = False


class AnthropicToolChoiceAuto(BaseModel):
    type: str = "auto"
    disable_parallel_tool_use: bool | None = False


ToolChoice = Union[
    Literal["none", "auto", "required", "any"], ToolFunctionChoice, AnthropicToolChoiceTool, AnthropicToolChoiceAny, AnthropicToolChoiceAuto
]


## tools ##
class FunctionSchema(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None  # JSON Schema for the parameters
    strict: bool = False


class Tool(BaseModel):
    # The type of the tool. Currently, only function is supported
    type: Literal["function"] = "function"
    # type: str = Field(default="function", const=True)
    function: FunctionSchema


## function_call ##
FunctionCallChoice = Union[Literal["none", "auto"], FunctionCall]


class ChatCompletionRequest(BaseModel):
    """https://platform.openai.com/docs/api-reference/chat/create"""

    model: str
    messages: list[ChatMessage | dict]
    frequency_penalty: float | None = 0
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = False
    top_logprobs: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = 1
    presence_penalty: float | None = 0
    response_format: ResponseFormat | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = False
    temperature: float | None = 1
    top_p: float | None = 1
    user: str | None = None  # unique ID of the end-user (for monitoring)
    parallel_tool_calls: bool | None = None
    instructions: str | None = None
    max_tokens: int | None = None

    # function-calling related
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None  # "none" means don't call a tool
    # deprecated scheme
    functions: list[FunctionSchema] | None = None
    function_call: FunctionCallChoice | None = None

    @field_validator("messages", mode="before")
    @classmethod
    def cast_all_messages(cls, v):
        return [cast_message_to_subtype(m) if isinstance(m, dict) else m for m in v]
