"""
V3 cache-layout helpers for AnthropicClient.

Pure functions — no client state needed. Kept separate so they can be
tested and reasoned about without pulling in the full client.

Layout when v3 is active (uses all 4 of Anthropic's ephemeral breakpoints):

    Block 1  [base instructions]                 → cache_control: ephemeral
    Block 2  [<persona>]                         → cache_control: ephemeral
    Block 3  [<human> + <conversation_summary>]  → cache_control: ephemeral
    Block 4  [tool_usage_rules + files]          → cache_control: ephemeral
    Block 5  [<memory_metadata>]                 (uncached — per-second timestamp)

A core_memory_append to human invalidates blocks 3-5 only; base and persona
remain cached. No messages-tail breakpoint (avoids the v2 cost regression).
"""

from __future__ import annotations


def split_system_message_v3(
    system_content: str,
) -> tuple[str, str, str, str, str] | None:
    """Split a Letta system prompt into five parts for v3 prompt caching.

    The system prompt is expected to contain Letta memory-block XML tags in
    this order: ``<persona>``, ``<human>``, ``<conversation_summary>``,
    ``</memory_blocks>``, and optionally ``<memory_metadata>``.

    The five parts map to the five cache blocks:

    - ``static_base``         — everything before ``<persona>`` (base instructions)
    - ``dynamic_persona``     — ``<persona>…</persona>``
    - ``dynamic_human_summary`` — ``<human>…</human>`` + ``<conversation_summary>…</conversation_summary>``
      up to and including ``</memory_blocks>`` when present
    - ``static_rules``        — tool_usage_rules / files section after the memory
      blocks and before ``<memory_metadata>``
    - ``dynamic_metadata``    — ``<memory_metadata>…`` to end of string, or ``""``
      when the tag is absent

    Args:
        system_content: The raw system-prompt string as assembled by the agent.

    Returns:
        A five-tuple ``(static_base, dynamic_persona, dynamic_human_summary,
        static_rules, dynamic_metadata)`` when all required markers are present,
        or ``None`` when any required marker is missing — callers should fall
        back to the v1 splitter in that case.
    """
    persona_start = system_content.find("<persona>")
    persona_end_idx = system_content.find("</persona>")
    conv_summary_end_idx = system_content.find("</conversation_summary>")
    memory_blocks_end_idx = system_content.find("</memory_blocks>")
    memory_metadata_start = system_content.find("<memory_metadata>")

    if persona_start == -1 or persona_end_idx == -1 or conv_summary_end_idx == -1:
        return None

    persona_end = persona_end_idx + len("</persona>")
    conv_summary_end = conv_summary_end_idx + len("</conversation_summary>")
    # include </memory_blocks> closing tag in block 3 if it follows </conversation_summary>
    if memory_blocks_end_idx != -1 and memory_blocks_end_idx > conv_summary_end:
        human_summary_end = memory_blocks_end_idx + len("</memory_blocks>")
    else:
        human_summary_end = conv_summary_end

    static_base = system_content[:persona_start].strip()
    dynamic_persona = system_content[persona_start:persona_end].strip()
    dynamic_human_summary = system_content[persona_end:human_summary_end].strip()

    if memory_metadata_start != -1 and memory_metadata_start > human_summary_end:
        static_rules = system_content[human_summary_end:memory_metadata_start].strip()
        dynamic_metadata = system_content[memory_metadata_start:].strip()
    else:
        static_rules = system_content[human_summary_end:].strip()
        dynamic_metadata = ""

    return static_base, dynamic_persona, dynamic_human_summary, static_rules, dynamic_metadata


def build_cache_blocks_v3(
    static_base: str,
    dynamic_persona: str,
    dynamic_human_summary: str,
    static_rules: str,
    dynamic_metadata: str,
) -> list[dict]:
    """Assemble Anthropic system-block dicts with v3 ``cache_control`` placement.

    Converts the five-part split produced by :func:`split_system_message_v3`
    into the list-of-dicts format expected by the Anthropic Messages API
    ``system`` field.

    Each non-empty block gets ``cache_control: {"type": "ephemeral"}`` except
    ``dynamic_metadata``, which is left uncached because the ``<memory_metadata>``
    section contains a per-second timestamp that changes on every turn and would
    defeat the cache if it were used as a breakpoint.

    Args:
        static_base: Base instructions before ``<persona>`` (block 1).
        dynamic_persona: ``<persona>`` block content (block 2).
        dynamic_human_summary: ``<human>`` + ``<conversation_summary>`` content (block 3).
        static_rules: Tool-usage rules and files section (block 4).
        dynamic_metadata: ``<memory_metadata>`` content, or ``""`` when absent (block 5).

    Returns:
        A list of Anthropic system-block dicts ready to assign to
        ``request_data["system"]``. Empty parts are omitted from the list.
    """
    parts: list[dict] = []
    if static_base:
        parts.append({"type": "text", "text": static_base, "cache_control": {"type": "ephemeral"}})
    if dynamic_persona:
        parts.append({"type": "text", "text": dynamic_persona, "cache_control": {"type": "ephemeral"}})
    if dynamic_human_summary:
        parts.append({"type": "text", "text": dynamic_human_summary, "cache_control": {"type": "ephemeral"}})
    if static_rules:
        parts.append({"type": "text", "text": static_rules, "cache_control": {"type": "ephemeral"}})
    if dynamic_metadata:
        parts.append({"type": "text", "text": dynamic_metadata})
    return parts
