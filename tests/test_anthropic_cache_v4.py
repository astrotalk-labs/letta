"""v4 history caching layout (cache_history_v4) for AnthropicClient.build_request_data.

Guards: nothing from the system prompt is dropped, at most 4 breakpoints, the
prefix through the history is byte-stable across turns even when memory and
memory_metadata change, mid-turn steps can read the previous step's prefix,
and flag-off / fallback paths are untouched.
"""

import copy
import json

import pytest

from letta.llm_api.anthropic_cache_v4 import MEMORY_POINTER, NOTE_HEADER, count_cache_breakpoints, split_system_message_v4
from letta.llm_api.anthropic_client import AnthropicClient
from letta.schemas.letta_message_content import TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message, ToolReturn
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall as ToolCall
from openai.types.chat.chat_completion_message_tool_call import Function as FunctionCall

BASE = (
    "You are Letta, the latest version of Limnal Corporation's digital companion.\n"
    "Your core memory unit is initialized with a <persona> chosen by the user, and a <human> block.\n" * 40
)
PERSONA = "<persona>\n<value>I am Pandit Ramesh, a warm Vedic astrologer from Varanasi.</value>\n</persona>"
RULES = "</memory_blocks>\n\n<tool_usage_rules>\nsend_message ends your turn.\n</tool_usage_rules>"


def system_prompt(human: str, summary: str, now: str, n_messages: int) -> str:
    return (
        f"{BASE}\n<memory_blocks>\nThe following memory blocks are currently engaged in your core memory unit:\n\n"
        f"<human>\n<value>{human}</value>\n</human>\n\n"
        f"<conversation_summary>\n<value>{summary}</value>\n</conversation_summary>\n\n"
        f"{PERSONA}\n\n{RULES}\n\n"
        f"<memory_metadata>\n- The current time is: {now}\n- {n_messages} previous messages\n</memory_metadata>"
    )


TOOLS = [
    {"name": "send_message", "description": "Send a message", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}},
    {"name": "core_memory_append", "description": "Append to memory", "parameters": {"type": "object", "properties": {"label": {"type": "string"}, "content": {"type": "string"}}, "required": ["label", "content"]}},
]


def llm_config(inner_thoughts_in_kwargs=True) -> LLMConfig:
    return LLMConfig(
        model="claude-sonnet-4-5-20250929",
        model_endpoint_type="anthropic_bedrock",
        context_window=200000,
        max_tokens=1024,
        put_inner_thoughts_in_kwargs=inner_thoughts_in_kwargs,
    )


def sys_msg(text):
    return Message(role="system", content=[TextContent(text=text)], agent_id="agent-1")


def user_msg(text):
    return Message(role="user", content=[TextContent(text=json.dumps({"type": "user_message", "message": text}))], agent_id="agent-1")


def event_msg(text):
    return Message(role="system", content=[TextContent(text=text)], agent_id="agent-1")


def tool_call_pair(call_id, name, args, result):
    assistant = Message(
        role="assistant",
        content=[TextContent(text=f"thinking about {name}")],
        tool_calls=[ToolCall(id=call_id, type="function", function=FunctionCall(name=name, arguments=json.dumps(args)))],
        agent_id="agent-1",
    )
    tool = Message(
        role="tool",
        content=[TextContent(text=json.dumps({"status": "OK", "message": result}))],
        tool_call_id=call_id,
        name=name,
        tool_returns=[ToolReturn(status="success")],
        agent_id="agent-1",
    )
    return [assistant, tool]


def turn(i):
    """One completed turn as persisted: user, memory append, send_message."""
    return [
        user_msg(f"question {i}: meri shaadi kab hogi?"),
        *tool_call_pair(f"call_{i}a", "core_memory_append", {"label": "human", "content": f"fact {i}"}, "None"),
        *tool_call_pair(f"call_{i}b", "send_message", {"message": f"answer {i}"}, "None"),
    ]


def build(history_turns, current, human="Asks about marriage.", summary="Earlier chat.", now="10:00:00", flag=True, inner=True, tools=TOOLS):
    client = AnthropicClient(at_user_id="1")
    client.cache_history_v4 = flag
    history = [m for i in range(history_turns) for m in turn(i)]
    persisted = [sys_msg(system_prompt(human, summary, now, len(history)))] + history
    client.history_message_count = len(persisted)
    return client.build_request_data(persisted + current, llm_config(inner), tools, None)


def strip_cc(obj):
    if isinstance(obj, dict):
        return {k: strip_cc(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, (list, tuple)):
        return [strip_cc(v) for v in obj]
    return obj


def flat_blocks(messages):
    out = []
    for m in messages:
        content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        for b in content:
            out.append((m["role"], b))
    return out


def prefix_through_last_breakpoint(data, nth_from_end=1):
    """Blocks (role, block) of the messages array up to and including the nth-from-last marked block."""
    blocks = flat_blocks(data["messages"])
    marked = [i for i, (_, b) in enumerate(blocks) if "cache_control" in b]
    end = marked[-nth_from_end]
    return strip_cc(blocks[: end + 1])


CURRENT = [event_msg("remaining_time: 300s"), user_msg("aur job kab lagegi?")]


@pytest.mark.parametrize("inner", [True, False])
def test_layout_keeps_everything_and_uses_four_breakpoints(inner):
    data = build(3, CURRENT, inner=inner)
    system_text = "\n\n".join(b["text"] for b in data["system"])
    last_user = [m for m in data["messages"] if m["role"] == "user"][-1]
    note = last_user["content"][-1]["text"]

    assert len(data["system"]) == 2 and all("cache_control" in b for b in data["system"])
    assert count_cache_breakpoints(data) == 4
    assert note.startswith("<event>" + NOTE_HEADER)
    assert "cache_control" not in last_user["content"][-1]
    # Stable parts stay in system; volatile parts only in the note.
    assert PERSONA in system_text and "<tool_usage_rules>" in system_text and MEMORY_POINTER in system_text
    for volatile in ("<human>\n<value>", "<conversation_summary>\n<value>", "The current time is: 10:00:00"):
        assert volatile not in system_text.split("<memory_blocks>", 1)[1]
        assert volatile in note
    # Nothing dropped: every line of the original system prompt is still sent once.
    original = system_prompt("Asks about marriage.", "Earlier chat.", "10:00:00", 15)
    sent = system_text + "\n" + note
    for line in original.splitlines():
        assert line in sent


def test_history_breakpoint_on_last_history_message():
    data = build(2, CURRENT)
    blocks = flat_blocks(data["messages"])
    marked = [i for i, (_, b) in enumerate(blocks) if "cache_control" in b]
    # bp3: the send_message tool result closing turn 1 (tool results merge into user messages).
    role, block = blocks[marked[0]]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "call_1b"
    # bp4: the last block of this turn, right before the note.
    assert marked[1] == len(blocks) - 2


@pytest.mark.parametrize("inner", [True, False])
def test_prefix_is_stable_across_turns_when_memory_changes(inner):
    t1 = build(3, CURRENT, human="Asks about marriage.", now="10:00:00", inner=inner)
    # Next turn: turn 3 got persisted, the human block was edited, time moved on.
    t2 = build(4, [event_msg("remaining_time: 240s"), user_msg("next?")], human="Asks about marriage. fact 3", now="10:02:13", inner=inner)
    assert t2["system"] == t1["system"]
    assert t2["tools"] == t1["tools"]
    cached = prefix_through_last_breakpoint(t1, nth_from_end=2)  # t1's bp3
    assert strip_cc(flat_blocks(t2["messages"]))[: len(cached)] == cached


def test_mid_turn_step_reads_previous_step_prefix():
    step1 = build(3, CURRENT)
    # Step 2 of the same turn: a memory append happened and rebuilt the system prompt.
    step2_current = CURRENT + tool_call_pair("call_x", "core_memory_append", {"label": "human", "content": "new"}, "None")
    step2 = build(3, step2_current, human="Asks about marriage. new", now="10:00:04")
    assert step2["system"] == step1["system"]
    cached = prefix_through_last_breakpoint(step1, nth_from_end=1)  # step1's bp4
    assert strip_cc(flat_blocks(step2["messages"]))[: len(cached)] == cached
    assert count_cache_breakpoints(step2) == 4


def test_first_turn_without_history():
    data = build(0, CURRENT)
    assert count_cache_breakpoints(data) == 3  # bp1, bp2, bp4
    assert data["messages"][-1]["content"][-1]["text"].startswith("<event>")


def test_flag_off_is_unchanged_v1():
    off = build(3, CURRENT, flag=False)
    v1 = AnthropicClient(at_user_id="1")._split_system_message_for_caching(system_prompt("Asks about marriage.", "Earlier chat.", "10:00:00", 15))
    assert [b["text"] for b in off["system"]] == [p for p in v1 if p]
    assert count_cache_breakpoints(off) == 2
    assert NOTE_HEADER not in json.dumps(off["messages"])


def test_v4_takes_precedence_over_v3():
    client = AnthropicClient(at_user_id="1")
    client.cache_history_v4 = True
    client.cache_optimisation_v3 = True
    history = [m for i in range(2) for m in turn(i)]
    persisted = [sys_msg(system_prompt("h", "s", "10:00:00", 6))] + history
    client.history_message_count = len(persisted)
    data = client.build_request_data(persisted + CURRENT, llm_config(), TOOLS, None)
    assert len(data["system"]) == 2 and count_cache_breakpoints(data) == 4


def test_falls_back_when_markers_missing_or_no_tools():
    client = AnthropicClient(at_user_id="1")
    client.cache_history_v4 = True
    plain = [sys_msg("You are a helpful assistant."), user_msg("hi")]
    client.history_message_count = 1
    data = client.build_request_data(plain, llm_config(), TOOLS, None)
    assert NOTE_HEADER not in json.dumps(data)
    # Summarizer path: no tools.
    data = build(2, CURRENT, tools=None)
    assert NOTE_HEADER not in json.dumps(data["messages"])


def test_split_requires_human_block():
    assert split_system_message_v4("no markers here") is None
    split = split_system_message_v4(system_prompt("h", "s", "t", 1))
    assert split is not None and split[2].startswith("<human>") and "<memory_metadata>" in split[2]
