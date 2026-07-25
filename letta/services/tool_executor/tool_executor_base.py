from abc import ABC, abstractmethod
from typing import Any

from letta.schemas.agent import AgentState
from letta.schemas.sandbox_config import SandboxConfig
from letta.schemas.tool import Tool
from letta.schemas.tool_execution_result import ToolExecutionResult
from letta.schemas.user import User
from letta.services.agent_manager import AgentManager
from letta.services.block_manager import BlockManager
from letta.services.message_manager import MessageManager
from letta.services.passage_manager import PassageManager


class ToolExecutor(ABC):
    """Abstract base class for tool executors."""

    def __init__(
        self,
        message_manager: MessageManager,
        agent_manager: AgentManager,
        block_manager: BlockManager,
        passage_manager: PassageManager,
        actor: User,
        task_id: str | None = None,
    ):
        self.message_manager = message_manager
        self.agent_manager = agent_manager
        self.block_manager = block_manager
        self.passage_manager = passage_manager
        self.actor = actor
        self.task_id = task_id

    @abstractmethod
    async def execute(
        self,
        function_name: str,
        function_args: dict,
        tool: Tool,
        actor: User,
        agent_state: AgentState | None = None,
        sandbox_config: SandboxConfig | None = None,
        sandbox_env_vars: dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        """Execute the tool and return the result."""
