from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class SystemMessage(BaseModel):
    content: str
    role: str = "system"
    name: str | None = None


class UserMessage(BaseModel):
    content: str | list[str]
    role: str = "user"
    name: str | None = None


class ToolCallFunction(BaseModel):
    name: str = Field(..., description="The name of the function to call")
    arguments: str = Field(..., description="The arguments to pass to the function (JSON dump)")


class ToolCall(BaseModel):
    id: str = Field(..., description="The ID of the tool call")
    type: str = "function"
    function: ToolCallFunction = Field(..., description="The arguments and name for the function")


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
    if role == "system":
        return SystemMessage(**m_dict)
    elif role == "user":
        return UserMessage(**m_dict)
    elif role == "assistant":
        return AssistantMessage(**m_dict)
    elif role == "tool":
        return ToolMessage(**m_dict)
    else:
        raise ValueError("Unknown message role")


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


ToolChoice = Union[Literal["none", "auto"], ToolFunctionChoice]


## tools ##
class FunctionSchema(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None  # JSON Schema for the parameters


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
    messages: list[ChatMessage]
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

    # function-calling related
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = "none"
    # deprecated scheme
    functions: list[FunctionSchema] | None = None
    function_call: FunctionCallChoice | None = None
