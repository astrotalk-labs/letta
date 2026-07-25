from typing import Any, Literal

from pydantic import BaseModel, Field

from letta.schemas.agent import AgentState


class ToolExecutionResult(BaseModel):

    status: Literal["success", "error"] = Field(..., description="The status of the tool execution and return object")
    func_return: Any | None = Field(None, description="The function return object")
    agent_state: AgentState | None = Field(None, description="The agent state")
    stdout: list[str] | None = Field(None, description="Captured stdout (prints, logs) from function invocation")
    stderr: list[str] | None = Field(None, description="Captured stderr from the function invocation")
    sandbox_config_fingerprint: str | None = Field(None, description="The fingerprint of the config for the sandbox")

    @property
    def success_flag(self) -> bool:
        return self.status == "success"
