"""
V3 cache-layout helpers for AnthropicClient.

Pure functions — no client state needed. Kept separate so they can be
tested and reasoned about without pulling in the full client.

## Cache block order (by change frequency, most-stable first)

The Letta system prompt's XML tags appear in text order:
  base → <human> → <conversation_summary> → <persona> → tool_usage_rules → files → memory_metadata

We EXTRACT each section and REASSEMBLE in cache-stability order so that the
most-stable content gets the lowest-indexed cache breakpoint:

    Block 1  [base instructions + <memory_blocks> header]  → cache_control: ephemeral
    Block 2  [<persona>]                                   → cache_control: ephemeral
    Block 3  [<human> + <conversation_summary>]            → cache_control: ephemeral
    Block 4  [</memory_blocks> + tool_usage_rules + files] → cache_control: ephemeral
    Block 5  [<memory_metadata>]                           (uncached — per-second timestamp)

Change frequency (observed / expected):
  base             → never changes within a session
  persona          → rarely changes (consultant personality)
  human            → changes on core_memory_append label="human"
  summary+rules    → changes on each summarization cycle
  metadata         → changes every second

The order can be tuned iteratively as observability logs reveal actual change
rates. See `_build_system_blocks_v3` in anthropic_client.py for the debug log
that surfaces per-block lengths each call.
"""

from __future__ import annotations

import hashlib


def split_system_message_v3(
    system_content: str,
) -> tuple[str, str, str, str, str] | None:
    """Extract five sections from *system_content* in cache-stability order.

    The raw system prompt stores memory blocks in text order
    ``<human> → <conversation_summary> → <persona>``.  This function extracts
    each section individually and returns them reordered so that the most
    stable content is first (lowest cache breakpoint index).

    Returned tuple fields:

    1. ``static_base``          — everything before ``<human>`` (base instructions
                                  and the ``<memory_blocks>`` wrapper header).
    2. ``dynamic_persona``      — ``<persona>…</persona>`` verbatim.
    3. ``dynamic_human``        — ``<human>…</human>`` verbatim.
    4. ``dynamic_summary_rules``— everything between ``</human>`` and
                                  ``<memory_metadata>`` with the ``<persona>``
                                  span excised (contains ``<conversation_summary>``,
                                  ``</memory_blocks>``, ``<tool_usage_rules>``,
                                  ``<files>``).
    5. ``dynamic_metadata``     — ``<memory_metadata>…`` to end of string, or
                                  ``""`` when the tag is absent.

    Returns ``None`` when ``<human>`` or ``<persona>`` markers are missing or
    out of order — the caller should fall back to the v1 splitter.
    ``<conversation_summary>`` is optional: agents without a summary yet
    produce an empty block 4 prefix, which is safe.
    """
    # The base instructions contain inline text references to <persona> and <human> that precede
    # the actual memory block tags.  Search for block tags only within the <memory_blocks> section
    # to skip those false matches.
    memory_blocks_pos = system_content.find("<memory_blocks>")
    search_from = memory_blocks_pos if memory_blocks_pos != -1 else 0

    human_start = system_content.find("<human>", search_from)
    persona_start = system_content.find("<persona>", search_from)
    persona_end_idx = system_content.find("</persona>", search_from)
    memory_metadata_start = system_content.find("<memory_metadata>")

    # Both <human> and <persona> are required.
    if human_start == -1 or persona_start == -1 or persona_end_idx == -1:
        return None

    # Basic ordering sanity: persona must close after it opens, and human must come before persona.
    if persona_end_idx <= persona_start or persona_start <= human_start:
        return None

    persona_end = persona_end_idx + len("</persona>")

    # Block 1: base + <memory_blocks> header — everything before <human> in natural text order.
    static_base = system_content[:human_start].strip()

    # Block 2: <persona>...</persona> — extracted and placed second (most stable memory block).
    dynamic_persona = system_content[persona_start:persona_end].strip()

    # Block 3: <human>...</human> + <conversation_summary>...</conversation_summary> together.
    # In natural text order these sit between <human> and <persona>, so we take that slice directly.
    # If <conversation_summary> is absent (new agent), this block is just <human>...</human>.
    dynamic_human_summary = system_content[human_start:persona_start].strip()

    # Block 4: </memory_blocks> + <tool_usage_rules> + <files> — everything after </persona>
    # up to <memory_metadata>.
    if memory_metadata_start != -1 and memory_metadata_start > persona_end:
        static_rules = system_content[persona_end:memory_metadata_start].strip()
        dynamic_metadata = system_content[memory_metadata_start:].strip()
    else:
        static_rules = system_content[persona_end:].strip()
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
    ``system`` field.  Content is **reordered** from its natural text position:
    persona is placed before human+summary so that the most stable memory block
    gets the lowest cache-breakpoint index.

    Blocks 1-4 each receive ``cache_control: {"type": "ephemeral"}``.
    Block 5 (``dynamic_metadata``) is left uncached — it contains a
    per-second timestamp that changes on every turn.

    Args:
        static_base: Base instructions + ``<memory_blocks>`` header (block 1).
        dynamic_persona: ``<persona>`` block (block 2, reordered to be stable).
        dynamic_human_summary: ``<human>`` + ``<conversation_summary>`` (block 3).
        static_rules: ``</memory_blocks>`` + tool_usage_rules + files (block 4).
        dynamic_metadata: ``<memory_metadata>`` content, or ``""`` (block 5).

    Returns:
        A list of Anthropic system-block dicts ready to assign to
        ``request_data["system"]``.  Empty parts are omitted.
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


def block_fingerprints(
    static_base: str,
    dynamic_persona: str,
    dynamic_human_summary: str,
    static_rules: str,
    dynamic_metadata: str,
) -> dict:
    """Return a dict of per-block lengths and 8-char content hashes for debug logs.

    Used in ``_build_system_blocks_v3`` to surface which blocks changed between
    turns, so the cache-stability order can be validated and tuned over time.
    """

    def _fp(s: str) -> str:
        return hashlib.md5(s.encode(), usedforsecurity=False).hexdigest()[:8]

    return {
        "b1_base_len": len(static_base),
        "b1_base_hash": _fp(static_base),
        "b2_persona_len": len(dynamic_persona),
        "b2_persona_hash": _fp(dynamic_persona),
        "b3_human_summary_len": len(dynamic_human_summary),
        "b3_human_summary_hash": _fp(dynamic_human_summary),
        "b4_rules_len": len(static_rules),
        "b4_rules_hash": _fp(static_rules),
        "b5_metadata_len": len(dynamic_metadata),
    }
