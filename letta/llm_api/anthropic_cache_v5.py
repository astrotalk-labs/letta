"""
V5/V6 cache-layout helpers for AnthropicClient.

Pure functions — no client state needed (same convention as v3/v4).

## v5 layout (3 system breakpoints)

Merges base + persona + tool_usage_rules + files into one stable block,
then gives <conversation_summary> and <human> their own breakpoints so that
a human memory edit only invalidates the last breakpoint:

    system   [base + persona + tool_usage_rules + files]  cache_control (bp1)
             [<conversation_summary>]                      cache_control (bp2)
             [<human>]                                     cache_control (bp3)
             [<memory_metadata>]                           uncached

Win vs v3: in v3, a <human> edit invalidates human+summary+tool_rules.
Here only bp3 (<human>) is invalidated; bp1 and bp2 are still cache hits.

## v6 layout (v5 system + 1 message history breakpoint)

Same 3 system breakpoints as v5. The freed 4th breakpoint slot is used
to cache the conversation history at the last complete 8-message boundary
before the current turn. Conversations of 9-16 msgs: bp at msg 8.
17-24 msgs: bp at msg 16 (most recent complete boundary).
"""

from __future__ import annotations

CACHE_CONTROL = {"type": "ephemeral"}


def split_system_message_v5(system_content: str) -> tuple[str, str, str, str] | None:
    """Extract four sections for the v5 cache layout.

    Returns:
        (stable_block, dynamic_summary, dynamic_human, dynamic_metadata)
        - stable_block: base + persona + tool_usage_rules + files (all merged)
        - dynamic_summary: <conversation_summary>...</conversation_summary>, or ""
        - dynamic_human: <human>...</human>
        - dynamic_metadata: <memory_metadata>... to end, or ""

    Returns None when the v3 splitter fails (missing required markers).
    When <conversation_summary> is absent (new agent), dynamic_summary is ""
    and dynamic_human holds the full human block.
    """
    from letta.llm_api.anthropic_cache_v3 import split_system_message_v3

    split = split_system_message_v3(system_content)
    if split is None:
        return None
    static_base, dynamic_persona, dynamic_human_summary, static_rules, dynamic_metadata = split

    # bp1: merge all stable parts — order: base → persona → rules+files
    stable_block = "\n\n".join(p for p in (static_base, dynamic_persona, static_rules) if p)

    # Try to split <conversation_summary> from <human>
    SUMMARY_TAG = "<conversation_summary>"
    SUMMARY_END_TAG = "</conversation_summary>"
    s_start = dynamic_human_summary.find(SUMMARY_TAG)
    if s_start == -1:
        # No summary yet (new agent) — keep full slice as human, no summary bp
        return stable_block, "", dynamic_human_summary, dynamic_metadata

    s_end = dynamic_human_summary.find(SUMMARY_END_TAG, s_start) + len(SUMMARY_END_TAG)
    dynamic_human = dynamic_human_summary[:s_start].strip()
    dynamic_summary = dynamic_human_summary[s_start:s_end].strip()
    return stable_block, dynamic_summary, dynamic_human, dynamic_metadata


def build_cache_blocks_v5(
    stable_block: str,
    dynamic_summary: str,
    dynamic_human: str,
    dynamic_metadata: str,
) -> list[dict]:
    """Assemble Anthropic system-block dicts for v5.

    3 breakpoints when summary present, 2 when absent (new agent):
      bp1: stable_block (base + persona + rules + files)
      bp2: dynamic_summary (skipped when empty)
      bp3: dynamic_human
      uncached: dynamic_metadata
    """
    parts: list[dict] = []
    if stable_block:
        parts.append({"type": "text", "text": stable_block, "cache_control": dict(CACHE_CONTROL)})
    if dynamic_summary:
        parts.append({"type": "text", "text": dynamic_summary, "cache_control": dict(CACHE_CONTROL)})
    if dynamic_human:
        parts.append({"type": "text", "text": dynamic_human, "cache_control": dict(CACHE_CONTROL)})
    if dynamic_metadata:
        parts.append({"type": "text", "text": dynamic_metadata})  # intentionally uncached
    return parts


def apply_single_history_breakpoint(messages: list[dict], interval: int = 8) -> bool:
    """Place one cache breakpoint at the last complete interval boundary before current turn.

    With only 1 message bp slot available (4 total - 3 used by v5 system), we mark
    the most recent complete 8-message boundary. This gives the model a stable history
    prefix that grows in 8-message chunks:
      - ≤8 msgs: no bp (not enough history)
      - 9-16 msgs: bp at msg 8
      - 17-24 msgs: bp at msg 16
      - etc.

    Returns True if a breakpoint was placed, False otherwise.
    """
    from letta.llm_api.anthropic_cache_v4 import mark_last_cacheable_block

    # Need at least `interval` history messages plus the current-turn message
    if len(messages) < interval + 1:
        return False

    # Last complete boundary index (0-based).
    # len-1 is the current-turn message; we don't touch it here.
    boundary_idx = ((len(messages) - 1) // interval) * interval - 1
    if boundary_idx < 0:
        return False

    return mark_last_cacheable_block(messages[boundary_idx])


def block_fingerprints_v5(
    stable_block: str,
    dynamic_summary: str,
    dynamic_human: str,
    dynamic_metadata: str,
) -> dict:
    """Return per-block lengths and 8-char hashes for debug logs."""
    import hashlib

    def _fp(s: str) -> str:
        return hashlib.md5(s.encode(), usedforsecurity=False).hexdigest()[:8]

    return {
        "b1_stable_len": len(stable_block),
        "b1_stable_hash": _fp(stable_block),
        "b2_summary_len": len(dynamic_summary),
        "b2_summary_hash": _fp(dynamic_summary),
        "b3_human_len": len(dynamic_human),
        "b3_human_hash": _fp(dynamic_human),
        "b4_metadata_len": len(dynamic_metadata),
    }
