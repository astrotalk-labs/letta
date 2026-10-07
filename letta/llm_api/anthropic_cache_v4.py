"""
V4 cache layout for AnthropicClient: cache the conversation history.

Pure functions — no client state needed (same convention as anthropic_cache_v3).

## Why history was never cached

v1 and v3 only cache the system prompt. Everything after it — the whole message
history — is billed as uncached input on every LLM call (~38% of Letta's Sonnet
input tokens, ~$68k/week). v2 tried a breakpoint on the last assistant message
and regressed cost +17%: the system prompt in front of the history changes on
almost every call (memory edits rebuild <human>, <memory_metadata> carries the
current time and message counts), so the history prefix was rewritten at 1.25x
instead of being read at 0.1x.

## What v4 does

It moves the parts of the system prompt that change turn to turn *after* the
history, so the prefix in front of the history is stable:

    system   [base instructions + <memory_blocks> header]        cache_control (bp1)
             [<persona> + pointer line + </memory_blocks> + tool rules + files]
                                                                  cache_control (bp2)
    messages [history ... last history message]                   cache_control (bp3)
             [this turn: user input, system events, tool calls]   cache_control (bp4)
             [memory note: <human> + <conversation_summary> + <memory_metadata>]

Memory edits mid-turn (core_memory_append from the agent loop or the tool
executor) now only change the note at the very end, so they no longer
invalidate anything cached. bp4 lets step k+1 of a multi-step turn read step
k's prefix. Only edits to <persona>, the base instructions, the tool rules or
the tool list invalidate the history cache.

Nothing is dropped: every character of the system prompt is still sent, either
in the system blocks or in the note (see `split_system_message_v4`).
"""

from __future__ import annotations

from letta.llm_api.anthropic_cache_v3 import split_system_message_v3

CACHE_CONTROL = {"type": "ephemeral"}

# Constant line left in the system prompt where <human> / <conversation_summary>
# used to sit, so the model knows they are still core memory blocks it can edit.
MEMORY_POINTER = (
    "(The <human> and <conversation_summary> memory blocks change often, so their current values are "
    "not repeated here: they are in the SYSTEM ALERT core-memory event at the end of the conversation. "
    "They are still part of your core memory and are edited with the same memory tools.)"
)

NOTE_HEADER = (
    "SYSTEM ALERT: Current values of your frequently changing core memory blocks and memory metadata. "
    "This is not a message from the user; never mention or quote it."
)

# Block types that may carry cache_control. Thinking blocks may not.
_CACHEABLE_BLOCK_TYPES = frozenset({"text", "tool_use", "tool_result", "image", "document"})


def split_system_message_v4(system_content: str) -> tuple[str, str, str] | None:
    """Split the system prompt into (static_base, stable_tail, volatile).

    * ``static_base`` — base instructions + ``<memory_blocks>`` header (bp1).
    * ``stable_tail`` — ``<persona>``, the pointer line, ``</memory_blocks>``,
      tool usage rules and files (bp2).
    * ``volatile``    — ``<human>`` + ``<conversation_summary>`` (and any other
      block between them and ``<persona>``) + ``<memory_metadata>``. Goes into
      the note after the conversation.

    Returns ``None`` when the v3 splitter can't find its markers; the caller
    then falls back to the existing layout.
    """
    split = split_system_message_v3(system_content)
    if split is None:
        return None
    static_base, persona, human_summary, rules, metadata = split
    if not human_summary:
        return None
    stable_tail = "\n\n".join(p for p in (persona, MEMORY_POINTER, rules) if p)
    volatile = "\n\n".join(p for p in (human_summary, metadata) if p)
    return static_base, stable_tail, volatile


def build_system_blocks_v4(static_base: str, stable_tail: str) -> list[dict]:
    parts: list[dict] = []
    if static_base:
        parts.append({"type": "text", "text": static_base, "cache_control": dict(CACHE_CONTROL)})
    if stable_tail:
        parts.append({"type": "text", "text": stable_tail, "cache_control": dict(CACHE_CONTROL)})
    return parts


def memory_note_text(volatile: str) -> str:
    return f"<event>{NOTE_HEADER}\n<memory_blocks>\n{volatile}\n</memory_blocks>\n</event>"


def _content_blocks(message: dict) -> list:
    content = message.get("content")
    if isinstance(content, list):
        return content
    blocks = [{"type": "text", "text": content if content is not None else ""}]
    message["content"] = blocks
    return blocks


def mark_last_cacheable_block(message: dict) -> bool:
    """Put cache_control on the last block of ``message`` that can carry it.

    Converts string content to a single text block (the API treats both forms
    the same). Returns False when no block can be marked.
    """
    blocks = _content_blocks(message)
    for i in range(len(blocks) - 1, -1, -1):
        block = blocks[i]
        if isinstance(block, dict) and block.get("type") in _CACHEABLE_BLOCK_TYPES:
            if block.get("type") == "text" and not block.get("text"):
                continue
            blocks[i] = {**block, "cache_control": dict(CACHE_CONTROL)}
            return True
    return False


def append_memory_note(messages: list[dict], volatile: str) -> bool:
    """Mark the end of this turn (bp4) and append the memory note to the final user message.

    Must run after merge_tool_results_into_user_messages and before any
    assistant prefill is appended. Returns False (and changes nothing) when the
    last message isn't a user message.
    """
    if not messages or messages[-1].get("role") != "user":
        return False
    last = messages[-1]
    mark_last_cacheable_block(last)
    _content_blocks(last).append({"type": "text", "text": memory_note_text(volatile)})
    return True


def count_cache_breakpoints(data: dict) -> int:
    n = 0
    system = data.get("system")
    if isinstance(system, list):
        n += sum(1 for b in system if isinstance(b, dict) and "cache_control" in b)
    for tool in data.get("tools") or []:
        if isinstance(tool, dict) and "cache_control" in tool:
            n += 1
    for m in data.get("messages") or []:
        content = m.get("content")
        if isinstance(content, list):
            n += sum(1 for b in content if isinstance(b, dict) and "cache_control" in b)
    return n


def strip_message_cache_control(messages: list[dict]) -> None:
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            m["content"] = [{k: v for k, v in b.items() if k != "cache_control"} if isinstance(b, dict) else b for b in content]
