from abc import ABC, abstractmethod
from typing import Any, List, Optional

import openai

from letta.constants import DEFAULT_MAX_STEPS
from letta.helpers import ToolRulesSolver
from letta.helpers.datetime_helpers import get_utc_time, get_utc_timestamp_ns, ns_to_ms
from letta.log import get_logger
from letta.schemas.agent import AgentState
from letta.schemas.enums import MessageStreamStatus
from letta.schemas.letta_message_content import TextContent
from letta.schemas.letta_response import LettaResponse
from letta.schemas.letta_stop_reason import LettaStopReason, StopReasonType
from letta.schemas.message import Message, MessageCreate, MessageUpdate
from letta.schemas.usage import LettaUsageStatistics
from letta.schemas.user import User
from letta.services.agent_manager import AgentManager
from letta.services.helpers.agent_manager_helper import compile_system_message
from letta.services.message_manager import MessageManager
from letta.services.passage_manager import PassageManager
from letta.utils import united_diff

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for AI agents, handling message management, tool execution,
    and context tracking.
    """

    def __init__(
        self,
        agent_id: str,
        # TODO: Make required once client refactor hits
        openai_client: Optional[openai.AsyncClient],
        message_manager: MessageManager,
        agent_manager: AgentManager,
        actor: User,
    ):
        self.agent_id = agent_id
        self.openai_client = openai_client
        self.message_manager = message_manager
        self.agent_manager = agent_manager
        # TODO: Pass this in
        self.passage_manager = PassageManager()
        self.actor = actor
        self.logger = get_logger(agent_id)
        self._task_id: Optional[str] = None  # set per-request by step(); used to gate latency logs

    @abstractmethod
    async def step(self, input_messages: List[MessageCreate], max_steps: int = DEFAULT_MAX_STEPS) -> LettaResponse:
        """
        Main execution loop for the agent.
        """
        raise NotImplementedError

    def pre_process_input_message(self, input_messages: List[MessageCreate]) -> Any:
        """
        Pre-process function to run on the input_message.
        """

        def get_content(message: MessageCreate) -> str:
            if isinstance(message.content, str):
                return message.content
            elif message.content and len(message.content) == 1 and isinstance(message.content[0], TextContent):
                return message.content[0].text
            else:
                return ""

        return [{"role": input_message.role.value, "content": get_content(input_message)} for input_message in input_messages]

    async def _rebuild_memory_async(
        self,
        in_context_messages: List[Message],
        agent_state: AgentState,
        tool_rules_solver: Optional[ToolRulesSolver] = None,
        num_messages: Optional[int] = None,  # storing these calculations is specific to the voice agent
        num_archival_memories: Optional[int] = None,
    ) -> List[Message]:
        """
        Async version of function above. For now before breaking up components, changes should be made in both places.
        """
        try:
            # [DB Call] loading blocks (modifies: agent_state.memory.blocks)
            _t_refresh = get_utc_timestamp_ns()
            await self.agent_manager.refresh_memory_async(agent_state=agent_state, actor=self.actor)
            _refresh_ms = ns_to_ms(get_utc_timestamp_ns() - _t_refresh)

            # TODO: This is a pretty brittle pattern established all over our code, need to get rid of this
            curr_system_message = in_context_messages[0]
            curr_memory_str = agent_state.memory.compile()
            curr_system_message_text = curr_system_message.content[0].text
            if curr_memory_str in curr_system_message_text:
                if self._task_id:
                    logger.info(
                        f"[REBUILD_TIMING] task_id={self._task_id} agent_id={agent_state.id} refresh_ms={_refresh_ms} changed=false"
                    )
                return in_context_messages

            memory_edit_timestamp = get_utc_time()

            # [DB Call] size of messages and archival memories
            # todo: blocking for now
            _t_counts = get_utc_timestamp_ns()
            if num_messages is None:
                num_messages = await self.message_manager.size_async(actor=self.actor, agent_id=agent_state.id)
            if num_archival_memories is None:
                num_archival_memories = await self.passage_manager.agent_passage_size_async(actor=self.actor, agent_id=agent_state.id)
            _counts_ms = ns_to_ms(get_utc_timestamp_ns() - _t_counts)

            new_system_message_str = compile_system_message(
                system_prompt=agent_state.system,
                in_context_memory=agent_state.memory,
                in_context_memory_last_edit=memory_edit_timestamp,
                previous_message_count=num_messages - len(in_context_messages),
                archival_memory_size=num_archival_memories,
                tool_rules_solver=tool_rules_solver,
            )

            diff = united_diff(curr_system_message_text, new_system_message_str)
            if len(diff) > 0:
                logger.debug(f"Rebuilding system with new memory...\nDiff:\n{diff}")

                # Cache-observability event: system prompt is being rebuilt because
                # something in agent_state.memory changed. This invalidates the
                # downstream cache breakpoints. Emit a structured log so we can grep
                # for the rate of these rebuilds in production and quantify how much
                # they contribute to overall cache misses. Gated to the cache_obs
                # sample so log volume stays bounded.
                _mr_at_user_id = getattr(self, "at_user_id", None)
                if _mr_at_user_id:
                    try:
                        from letta.llm_api.anthropic_client import _is_user_in_cache_obs_sample

                        if _is_user_in_cache_obs_sample(_mr_at_user_id):
                            import json as _json

                            logger.info(
                                "[MEMORY_REBUILD] %s",
                                _json.dumps(
                                    {
                                        "at_user_id": _mr_at_user_id,
                                        "agent_id": agent_state.id,
                                        "diff_chars": len(diff),
                                        "old_system_chars": len(curr_system_message_text),
                                        "new_system_chars": len(new_system_message_str),
                                    },
                                    default=str,
                                ),
                            )
                    except Exception:
                        pass

                _step_idx = getattr(self, "_current_step_index", None)
                if _step_idx is not None and _step_idx > 0:
                    try:
                        import json as _json

                        logger.warning(
                            "[COST_LEAK] %s",
                            _json.dumps(
                                {
                                    "type": "mid_turn_rebuild",
                                    "at_user_id": _mr_at_user_id,
                                    "agent_id": agent_state.id,
                                    "step_index": _step_idx,
                                    "diff_chars": len(diff),
                                    "old_system_chars": len(curr_system_message_text),
                                },
                                default=str,
                            ),
                        )
                    except Exception:
                        pass

                # Test-user-gated optimization: skip system message rewrite when
                # the diff is small AND the total system length is unchanged. This
                # is the signature of pure attribute-only updates (chars_current,
                # memory_edit_timestamp) — same length, tiny char churn, no semantic
                # change. PR #59 obs data: ~half of MEMORY_REBUILDs have diff_chars
                # in 627-631 with old_chars == new_chars. The regex-strip approach
                # in the previous attempt didn't fire in production because the
                # diff format varies; the threshold check is format-independent.
                # Gated to test user 92744418 first; follow-up PR graduates.
                SKIP_TRIVIAL_REBUILD_USER_ID = "92744418"
                SKIP_TRIVIAL_REBUILD_MAX_DIFF = 700
                if (
                    _mr_at_user_id == SKIP_TRIVIAL_REBUILD_USER_ID
                    and len(diff) < SKIP_TRIVIAL_REBUILD_MAX_DIFF
                    and len(curr_system_message_text) == len(new_system_message_str)
                ):
                    try:
                        import json as _json

                        logger.info(
                            "[MEMORY_REBUILD_SKIPPED] %s",
                            _json.dumps(
                                {
                                    "at_user_id": _mr_at_user_id,
                                    "agent_id": agent_state.id,
                                    "diff_chars": len(diff),
                                    "system_chars": len(curr_system_message_text),
                                    "reason": "small_diff_equal_length",
                                },
                                default=str,
                            ),
                        )
                    except Exception:
                        pass
                    return in_context_messages

                # [DB Call] Update Messages
                _t_write = get_utc_timestamp_ns()
                new_system_message = await self.message_manager.update_message_by_id_async(
                    curr_system_message.id, message_update=MessageUpdate(content=new_system_message_str), actor=self.actor
                )
                _write_ms = ns_to_ms(get_utc_timestamp_ns() - _t_write)
                if self._task_id:
                    logger.info(
                        f"[REBUILD_TIMING] task_id={self._task_id} agent_id={agent_state.id} refresh_ms={_refresh_ms} "
                        f"counts_ms={_counts_ms} sysmsg_write_ms={_write_ms} changed=true"
                    )
                return [new_system_message] + in_context_messages[1:]

            else:
                if self._task_id:
                    logger.info(
                        f"[REBUILD_TIMING] task_id={self._task_id} agent_id={agent_state.id} refresh_ms={_refresh_ms} "
                        f"counts_ms={_counts_ms} sysmsg_write_ms=0 changed=false_nodiff"
                    )
                return in_context_messages
        except:
            logger.exception(f"Failed to rebuild memory for agent id={agent_state.id} and actor=({self.actor.id}, {self.actor.name})")
            raise
