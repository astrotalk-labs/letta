import asyncio
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import anthropic
from openai import AzureOpenAI

from letta.agents.base_agent import BaseAgent
from letta.constants import DEFAULT_MAX_STEPS
from letta.log import get_logger
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

logger = get_logger(__name__)

AZURE_SUMMARIZER_ROLLOUT_PERCENT = 10  # % of users routed to Azure


class EphemeralSummaryAgent(BaseAgent):
    """
    A stateless summarization agent using Azure OpenAI in a thread pool.
    """

    def __init__(
        self,
        target_block_label: str,
        agent_id: str,
        message_manager: MessageManager,
        agent_manager: AgentManager,
        block_manager: BlockManager,
        actor: User,
        at_user_id: Optional[str] = None,
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
        self.at_user_id = at_user_id

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

        use_azure = self.at_user_id and self.at_user_id.isdigit() and int(self.at_user_id) % 10 == 0

        if use_azure:
            logger.warning(
                    f"[SUMMARIZER] Using Azure OpenAI for summarization | at_user_id={self.at_user_id} | "
                f"agent_id={self.agent_id} | deployment={model_settings.letta_embedding_5_4_mini_deployment} | "
                f"endpoint={model_settings.letta_embedding_5_4_mini_base_url}"
            )

            def _invoke():
                client = AzureOpenAI(
                    api_key=model_settings.letta_embedding_5_4_mini_api_key,
                    api_version=model_settings.letta_embedding_5_4_mini_api_version,
                    azure_endpoint=model_settings.letta_embedding_5_4_mini_base_url,
                )
                return client.chat.completions.create(
                    model=model_settings.letta_embedding_5_4_mini_deployment,
                    max_completion_tokens=1500,
                    messages=[{"role": "system", "content": system}] + messages,
                )

            response = await asyncio.to_thread(_invoke)
            summary = response.choices[0].message.content.strip()
        else:
            logger.warning(
                f"[SUMMARIZER] Using Anthropic for summarization | at_user_id={self.at_user_id} | agent_id={self.agent_id}"
            )

            def _invoke():
                client = anthropic.Anthropic(api_key=model_settings.anthropic_api_key)
                return client.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=1500,
                    system=system,
                    messages=messages,
                )

            response = await asyncio.to_thread(_invoke)
            summary = response.content[0].text.strip()

        logger.warning(f"[SUMMARIZER] Summarization completed | at_user_id={self.at_user_id} | agent_id={self.agent_id} | provider={'azure' if use_azure else 'anthropic'} | summary_length={len(summary)}")

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
