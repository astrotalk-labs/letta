from enum import Enum

from pydantic import BaseModel, Field


class ImageFile(BaseModel):
    type: str = "image_file"
    file_id: str


class Text(BaseModel):
    object: str = "text"
    text: str = Field(..., description="The text content to be processed by the agent.")


class MessageRoleType(str, Enum):
    user = "user"
    system = "system"


class OpenAIAssistant(BaseModel):
    """Represents an OpenAI assistant (equivalent to Letta preset)"""

    id: str = Field(..., description="The unique identifier of the assistant.")
    name: str = Field(..., description="The name of the assistant.")
    object: str = "assistant"
    description: str | None = Field(None, description="The description of the assistant.")
    created_at: int = Field(..., description="The unix timestamp of when the assistant was created.")
    model: str = Field(..., description="The model used by the assistant.")
    instructions: str = Field(..., description="The instructions for the assistant.")
    tools: list[str] | None = Field(None, description="The tools used by the assistant.")
    file_ids: list[str] | None = Field(None, description="List of file IDs associated with the assistant.")
    metadata: dict | None = Field(None, description="Metadata associated with the assistant.")


class OpenAIMessage(BaseModel):
    id: str = Field(..., description="The unique identifier of the message.")
    object: str = "thread.message"
    created_at: int = Field(..., description="The unix timestamp of when the message was created.")
    thread_id: str = Field(..., description="The unique identifier of the thread.")
    role: str = Field(..., description="Role of the message sender (either 'user' or 'system')")
    content: list[Text | ImageFile] = Field(None, description="The message content to be processed by the agent.")
    assistant_id: str = Field(..., description="The unique identifier of the assistant.")
    run_id: str | None = Field(None, description="The unique identifier of the run.")
    file_ids: list[str] | None = Field(None, description="List of file IDs associated with the message.")
    metadata: dict | None = Field(None, description="Metadata associated with the message.")


class OpenAIThread(BaseModel):
    """Represents an OpenAI thread (equivalent to Letta agent)"""

    id: str = Field(..., description="The unique identifier of the thread.")
    object: str = "thread"
    created_at: int = Field(..., description="The unix timestamp of when the thread was created.")
    metadata: dict = Field(None, description="Metadata associated with the thread.")


class AssistantFile(BaseModel):
    id: str = Field(..., description="The unique identifier of the file.")
    object: str = "assistant.file"
    created_at: int = Field(..., description="The unix timestamp of when the file was created.")
    assistant_id: str = Field(..., description="The unique identifier of the assistant.")


class MessageFile(BaseModel):
    id: str = Field(..., description="The unique identifier of the file.")
    object: str = "thread.message.file"
    created_at: int = Field(..., description="The unix timestamp of when the file was created.")
    message_id: str = Field(..., description="The unique identifier of the message.")


class Function(BaseModel):
    name: str = Field(..., description="The name of the function.")
    arguments: str = Field(..., description="The arguments of the function.")


class ToolCall(BaseModel):
    id: str = Field(..., description="The unique identifier of the tool call.")
    type: str = "function"
    function: Function = Field(..., description="The function call.")


class ToolCallOutput(BaseModel):
    tool_call_id: str = Field(..., description="The unique identifier of the tool call.")
    output: str = Field(..., description="The output of the tool call.")


class RequiredAction(BaseModel):
    type: str = "submit_tool_outputs"
    submit_tool_outputs: list[ToolCall]


class OpenAIError(BaseModel):
    code: str = Field(..., description="The error code.")
    message: str = Field(..., description="The error message.")


class OpenAIUsage(BaseModel):
    completion_tokens: int = Field(..., description="The number of tokens used for the run.")
    prompt_tokens: int = Field(..., description="The number of tokens used for the prompt.")
    total_tokens: int = Field(..., description="The total number of tokens used for the run.")


class OpenAIMessageCreationStep(BaseModel):
    type: str = "message_creation"
    message_id: str = Field(..., description="The unique identifier of the message.")


class OpenAIToolCallsStep(BaseModel):
    type: str = "tool_calls"
    tool_calls: list[ToolCall] = Field(..., description="The tool calls.")


class OpenAIRun(BaseModel):
    id: str = Field(..., description="The unique identifier of the run.")
    object: str = "thread.run"
    created_at: int = Field(..., description="The unix timestamp of when the run was created.")
    thread_id: str = Field(..., description="The unique identifier of the thread.")
    assistant_id: str = Field(..., description="The unique identifier of the assistant.")
    status: str = Field(..., description="The status of the run.")
    required_action: RequiredAction | None = Field(None, description="The required action of the run.")
    last_error: OpenAIError | None = Field(None, description="The last error of the run.")
    expires_at: int = Field(..., description="The unix timestamp of when the run expires.")
    started_at: int | None = Field(None, description="The unix timestamp of when the run started.")
    cancelled_at: int | None = Field(None, description="The unix timestamp of when the run was cancelled.")
    failed_at: int | None = Field(None, description="The unix timestamp of when the run failed.")
    completed_at: int | None = Field(None, description="The unix timestamp of when the run completed.")
    model: str = Field(..., description="The model used by the run.")
    instructions: str = Field(..., description="The instructions for the run.")
    tools: list[ToolCall] | None = Field(None, description="The tools used by the run.")  # TODO: also add code interpreter / retrieval
    file_ids: list[str] | None = Field(None, description="List of file IDs associated with the run.")
    metadata: dict | None = Field(None, description="Metadata associated with the run.")
    usage: OpenAIUsage | None = Field(None, description="The usage of the run.")


class OpenAIRunStep(BaseModel):
    id: str = Field(..., description="The unique identifier of the run step.")
    object: str = "thread.run.step"
    created_at: int = Field(..., description="The unix timestamp of when the run step was created.")
    assistant_id: str = Field(..., description="The unique identifier of the assistant.")
    thread_id: str = Field(..., description="The unique identifier of the thread.")
    run_id: str = Field(..., description="The unique identifier of the run.")
    type: str = Field(..., description="The type of the run step.")  # message_creation, tool_calls
    status: str = Field(..., description="The status of the run step.")
    step_defaults: OpenAIToolCallsStep | OpenAIMessageCreationStep = Field(..., description="The step defaults.")
    last_error: OpenAIError | None = Field(None, description="The last error of the run step.")
    expired_at: int | None = Field(None, description="The unix timestamp of when the run step expired.")
    failed_at: int | None = Field(None, description="The unix timestamp of when the run failed.")
    completed_at: int | None = Field(None, description="The unix timestamp of when the run completed.")
    usage: OpenAIUsage | None = Field(None, description="The usage of the run.")
