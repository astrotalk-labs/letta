from pathlib import Path
from typing import AsyncGenerator, List

import httpx
import anthropic

from letta.agents.base_agent import BaseAgent
from letta.constants import DEFAULT_MAX_STEPS
from letta.orm.errors import NoResultFound
from letta.schemas.block import Block, BlockUpdate
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import TextContent
from letta.schemas.message import Message, MessageCreate
from letta.schemas.user import User
from letta.services.agent_manager import AgentManager
from letta.services.block_manager import BlockManager
from letta.services.message_manager import MessageManager
from letta.settings import model_settings

SUMMARIZER_MODEL = "claude-haiku-4-5"


class EphemeralSummaryAgent(BaseAgent):
    """
    A stateless summarization agent using Anthropic client directly.
    """

    def __init__(
        self,
        target_block_label: str,
        agent_id: str,
        message_manager: MessageManager,
        agent_manager: AgentManager,
        block_manager: BlockManager,
        actor: User,
    ):
        super().__init__(
            agent_id=agent_id,
            openai_client=None,
            message_manager=message_manager,
            agent_manager=agent_manager,
            actor=actor,
        )
        self.target_block_label = target_block_label
        self.block_manager = block_manager
        self.anthropic_client = anthropic.AsyncAnthropic(
            api_key=model_settings.anthropic_api_key,
            timeout=httpx.Timeout(timeout=60.0, connect=30.0),
        )

    async def step(self, input_messages: List[MessageCreate], max_steps: int = DEFAULT_MAX_STEPS) -> List[Message]:
        if len(input_messages) > 1:
            raise ValueError("Can only invoke EphemeralSummaryAgent with a single summarization message.")

        # Check block existence
        try:
            block = await self.agent_manager.get_block_with_label_async(
                agent_id=self.agent_id, block_label=self.target_block_label, actor=self.actor
            )
        except NoResultFound:
            block = await self.block_manager.create_or_update_block_async(
                block=Block(
                    value="", label=self.target_block_label, description="Contains recursive summarizations of the conversation so far"
                ),
                actor=self.actor,
            )
            await self.agent_manager.attach_block_async(agent_id=self.agent_id, block_id=block.id, actor=self.actor)

        if block.value:
            input_message = input_messages[0]
            input_message.content[0].text += f"\n\n--- Previous Summary ---\n{block.value}\n"

        messages = self.pre_process_input_message(input_messages=input_messages)

        current_dir = Path(__file__).parent
        with open(current_dir / "prompts" / "summary_system_prompt.txt", "r") as f:
            system = f.read()

        response = await self.anthropic_client.messages.create(
            model=SUMMARIZER_MODEL,
            max_tokens=4096,
            system=system,
            messages=messages,
        )
        summary = response.content[0].text.strip()

        await self.block_manager.update_block_async(block_id=block.id, block_update=BlockUpdate(value=summary), actor=self.actor)

        print(block)
        print(summary)

        return [
            Message(
                role=MessageRole.assistant,
                content=[TextContent(text=summary)],
            )
        ]

    async def step_stream(self, input_messages: List[MessageCreate], max_steps: int = DEFAULT_MAX_STEPS) -> AsyncGenerator[str, None]:
        raise NotImplementedError("EphemeralAgent does not support async step.")
