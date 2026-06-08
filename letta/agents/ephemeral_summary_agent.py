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

AZURE_SUMMARIZER_ROLLOUT_PERCENT = 0  # % of users routed to Azure


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

        # Cache-observability event: the summarizer is firing. When it runs, it
        # overwrites the <conversation_summary> block AND trims in_context_messages,
        # both of which invalidate the message-tail cache_control breakpoint on the
        # next user turn. Emit a structured log so we can measure how often this
        # happens in production and decide whether to decouple the summary block.
        # Always log (summarizer is rare, not per-call) but include at_user_id so
        # we can filter to the cache_obs sample if log volume becomes an issue.
        try:
            import json as _json
            _input_text = input_messages[0].content[0].text if input_messages and input_messages[0].content else ""
            logger.info(
                "[SUMMARIZER_FIRED] %s",
                _json.dumps({
                    "at_user_id": getattr(self, "at_user_id", None),
                    "agent_id": self.agent_id,
                    "target_block_label": self.target_block_label,
                    "input_text_chars": len(_input_text),
                }, default=str),
            )
        except Exception:
            pass

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

        use_azure = False

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
                # Cache the ~11k-token summarizer system prompt. SUMMARIZER_FIRED logs
                # show this code path runs roughly 1.5 times/second in production, each
                # call previously paying full price for the entire system prompt. With
                # cache_control: ephemeral and the prompt-caching-2024-07-31 beta, the
                # same prompt becomes a cache_read at 0.1x cost on subsequent calls
                # (within the 5-minute TTL). System prompt is fully static (loaded
                # from prompts/summary_system_prompt.txt) so it caches across all
                # summarizer invocations regardless of which agent triggered them.
                return client.beta.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=1500,
                    system=[
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=messages,
                    betas=["prompt-caching-2024-07-31"],
                )

            response = await asyncio.to_thread(_invoke)
            summary = response.content[0].text.strip()

            # Cache-observability for the summarizer. PR #60 enabled cache_control
            # on the system prompt but the dashboard impact was smaller than sized.
            # We don't know if cache_read is hitting — log usage from response so we
            # can see ground truth. Sampled to 1% (plus test user) via the existing
            # helper to bound log volume. Includes a hash of the messages array and
            # system prompt to spot non-determinism that would invalidate the cache.
            try:
                from letta.llm_api.anthropic_client import _is_user_in_cache_obs_sample
                if _is_user_in_cache_obs_sample(self.at_user_id):
                    import hashlib as _hashlib
                    import json as _json
                    usage = getattr(response, "usage", None)
                    _msgs_serialized = _json.dumps(messages, sort_keys=True, default=str)
                    _msgs_hash = _hashlib.sha256(_msgs_serialized.encode()).hexdigest()[:12]
                    _system_hash = _hashlib.sha256(system.encode()).hexdigest()[:12]
                    logger.info(
                        "[SUMMARIZER_USAGE] %s",
                        _json.dumps({
                            "at_user_id": self.at_user_id,
                            "agent_id": self.agent_id,
                            "input_tokens": getattr(usage, "input_tokens", None),
                            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
                            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                            "output_tokens": getattr(usage, "output_tokens", None),
                            "system_chars": len(system),
                            "system_hash": _system_hash,
                            "messages_chars": len(_msgs_serialized),
                            "messages_hash": _msgs_hash,
                            "num_messages": len(messages),
                        }, default=str),
                    )
            except Exception:
                pass

        logger.warning(f"[SUMMARIZER] Summarization completed | at_user_id={self.at_user_id} | agent_id={self.agent_id} | provider={'azure' if use_azure else 'anthropic'} | summary_length={len(summary)}")

        if block.limit and len(summary) > block.limit:
            truncated = summary[: block.limit]
            last_newline = truncated.rfind("\n")
            summary = truncated[:last_newline] if last_newline != -1 else truncated
            logger.warning(
                f"[SUMMARIZER] Summary truncated to {len(summary)} chars (limit={block.limit}) | "
                f"at_user_id={self.at_user_id} | agent_id={self.agent_id}"
            )

        await self.block_manager.update_block_async(block_id=block.id, block_update=BlockUpdate(value=summary), actor=self.actor)

        return [
            Message(
                role=MessageRole.assistant,
                content=[TextContent(text=summary)],
            )
        ]

    async def step_stream(self, input_messages: List[MessageCreate], max_steps: int = DEFAULT_MAX_STEPS) -> AsyncGenerator[str, None]:
        raise NotImplementedError("EphemeralAgent does not support async step.")
