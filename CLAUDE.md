# CLAUDE.md — Letta Server

Guidance for Claude Code when working in this repo (the Letta agent server itself,
Python/FastAPI). This file documents the REST flows that external callers (e.g. the
AstroTalk backend) exercise most heavily, so changes to these paths are made with
the full request lifecycle in mind — plus repo-specific coding conventions.

## Repo shape

- `letta/server/rest_api/routers/v1/*.py` — FastAPI route handlers (the public API surface).
- `letta/schemas/*.py` — Pydantic request/response models.
- `letta/agents/letta_agent.py` — `LettaAgent`, the main agent step loop.
- `letta/llm_api/*.py` — per-provider LLM clients (`anthropic_client.py`, `llm_client.py` factory).
- `letta/server/server.py` — `SyncServer`, the service-layer facade routers call into.

## Core external flows

An external caller typically drives an agent through this lifecycle. Endpoints below
are grep-verified against `letta/server/rest_api/routers/v1/`.

### 1. Create an agent — `POST /v1/agents/`

`agents.py::create_agent`, body is `CreateAgentRequest` (`letta/schemas/agent.py`).
`agent_type` is effectively always `"memgpt_agent"` for standard chat agents (other
types like `voice_convo_agent` exist for specialized flows — check `AgentType` before
assuming). Key fields: `name`, `system`, `llm_config`, `embedding_config`,
`memory_blocks` (list of `{value, label, limit}` — conventionally `persona` + `human`),
`tools`, `tool_rules`.

`llm_config.model_endpoint_type` is one of `"anthropic"`, `"anthropic_vertex"`,
`"anthropic_bedrock"`, `"openai"`, `"google_ai"`, `"google_vertex"`, etc. This value is
what `_apply_provider_switching` (below) mutates at request time, and what
`LLMClient.create()` (`letta/llm_api/llm_client.py`) reads to pick a client class.

### 2. Fetch / patch agent state

- `GET /v1/agents/{agent_id}` — full `AgentState`, including resolved memory blocks.
- `PATCH /v1/agents/{agent_id}` — body is `UpdateAgent`; used to swap `llm_config`
  (e.g. reverting a provider experiment back to direct Anthropic), or patch `system`.
- `PATCH /v1/agents/{agent_id}/tools/attach/{tool_id}` / `.../detach/{tool_id}` — tool
  wiring is via dedicated attach/detach routes, not a bare `tools` field on `UpdateAgent`.
  A caller sending `{"tools": [...]}` in a general PATCH is not exercising a real
  endpoint contract — if you see that pattern from a consumer, point them at the
  attach/detach routes or a dedicated tool-creation call instead.

### 3. Memory blocks

- `GET/PATCH /v1/agents/{agent_id}/core-memory/blocks/{block_label}` — read/update a
  block by label (`persona`, `human`, etc.), scoped to one agent.
- `PATCH /v1/blocks/{block_id}` (`blocks.py::modify_block`) — update a block directly
  by its own ID, independent of any agent. Body is `{"value": "..."}` at minimum
  (`BlockUpdate` schema) — this is the one to use when the caller already holds a
  `block_id` rather than an `(agent_id, label)` pair, since a block can be shared
  across agents.

### 4. Send a message — `POST /v1/agents/{agent_id}/messages` (the hot path)

`agents.py::send_message`, body is `LettaRequest` (`letta/schemas/letta_request.py`).
This is the highest-traffic, latency-sensitive endpoint — most perf/observability work
in this repo touches this handler and `LettaAgent.step()`.

Request fields that matter beyond `messages`:

| Field | Purpose |
|---|---|
| `use_vertex_experiment` / `use_bedrock_experiment` | Dynamic provider routing for this request only (see below). |
| `model_override` | Swap model for this request without touching the stored agent config (A/B testing). |
| `llm_provider` | Force `"google"` routing regardless of agent config. |
| `user_cohort` | Drives API-key routing (`AT_NATIVE`, `AT_FOREIGN`, `INDIAN_AT`, `PANDITJI`, `LUMUS`, `UNKNOWN`). |
| `business_id` | Raw business ID from the order (1=AstroTalk, 10=Panditji, 12=Lumus). |
| `thinking` | Extended-thinking config forwarded to Claude (forces `temperature=1`, `max_tokens>=8000` when set). |
| `output_config` | `{"effort": "low"\|"medium"\|"high"}` — for models that reject `budget_tokens` (e.g. `claude-sonnet-5`) and want effort-based reasoning instead. |
| `thinking_config` | Gemini-side thinking budget (`{"level": ...}`), separate from `thinking`. |
| `task_id` | Enables per-phase latency warning logs (context prep / LLM call / tool exec / context rebuild). |
| `latencyOptimisationFlow` (alias `latency_optimisation_flow`) | Routes memory-tool follow-up steps (`archival_memory_search`, `recall_memory_search`, `core_memory_append`, `core_memory_replace`, `archival_memory_insert`) to Haiku instead of the agent's main model. Step 0 and `send_message` always use the full model. Works across Anthropic direct / Vertex / Bedrock. |

**Dynamic provider switching** — `LettaAgent._apply_provider_switching()`
(`letta/agents/letta_agent.py:221`) mutates `agent_state.llm_config.model_endpoint_type`
at runtime based on `use_vertex_experiment` / `use_bedrock_experiment`, **before**
`LLMClient.create()` picks a client. `LettaAgent` also exposes
`step_stream_no_tokens()` / `step_stream()` with their own `_apply_provider_switching`
call sites (lines ~307/328, ~1229/1258), but **streaming isn't used by this
integration** — `POST /messages` calls `step()` (line ~261) exclusively. Don't spend
verification effort on the streaming call sites for provider-switching changes tied
to this flow; `LLMClient`'s factory falls back to `AnthropicClient` for both
`"anthropic_vertex"` and `"anthropic_bedrock"` endpoint types either way. (The
streaming methods do still exist in the codebase and could matter for other
integrations/endpoints — `POST /messages/stream` — so don't delete or neglect them
outright, just don't treat them as in-scope for this caller's request path.)

Routing inside the handler: if the agent is `multi_agent_group`-eligible and its
`model_endpoint_type` is one of `anthropic`/`openai`/`together`/`google_ai`/
`google_vertex`, it goes through `LettaAgent` (or `SleeptimeMultiAgentV2` if
`enable_sleeptime`); otherwise it falls back to the legacy
`server.send_message_to_agent(...)` path, which does **not** accept `thinking`,
`output_config`, `task_id`, or `latencyOptimisationFlow` — those are new-loop-only.
Keep this in mind when a caller reports a flag being silently ignored: check which
branch their agent actually takes first.

### 5. Archival memory — `POST /v1/agents/{agent_id}/archival-memory`

`agents.py::create_passage`, body is `{"text": "..."}` (list of `Passage` returned).
Callers that retry on failure should use bounded retries with backoff — this endpoint
does synchronous embedding generation, so a hot retry loop directly compounds
embedding-provider load during an outage rather than backing off from it.

## Best practices for this repo

- **`step_stream_no_tokens()` and `step_stream()` are dead code paths.**
  They are never invoked by any caller in this integration. All `/messages` traffic
  goes through `step()` → `_step()`. Do not spend effort verifying, fixing, or
  maintaining these methods — treat them as dead code.
- **Provider-affecting changes for the `/messages` flow only need `_step()`.**
  `use_vertex_experiment` / `use_bedrock_experiment` / `model_override` /
  `llm_provider` handling only needs to be verified in `LettaAgent._step()`.
- **New `LettaRequest` fields need explicit plumbing, not implicit inheritance.**
  `LettaStreamingRequest` and `LettaBatchRequest` subclass `LettaRequest`, but the
  *handlers* for streaming/batch call their own code paths — a field added to
  `LettaRequest` is not automatically read by the streaming or batch handler. Trace
  the field from schema → router → `LettaAgent` method signature → actual use.
- **Match camelCase/snake_case aliasing intentionally.** `latencyOptimisationFlow`
  uses `AliasChoices` to accept both cases because external callers are inconsistent.
  If you add a similarly caller-facing field, decide up front whether both cases need
  support, and use `AliasChoices` rather than silently accepting only one.
- **This is a latency-sensitive endpoint.** `/messages` records an E2E histogram
  (`MetricRegistry().messages_endpoint_e2e_ms_histogram`) tagged by
  `latency_optimisation_flow` and `status_code` in a `finally` block. Any change that
  adds work to the request path (new provider check, new logging, new metric) should
  consider whether it belongs before or after this timer starts, and whether it's
  conditional on a flag rather than always-on.
- **Format with `black` (line profile: isort profile=`black`) before submitting.**
  Run `poetry run pre-commit run --all-files`, or at minimum `poetry run black .` and
  `poetry run isort .`, before a PR — this repo enforces it in CI.
- **Tests**: `pytest-asyncio` is used throughout for the async agent/server code paths
  — new tests for async handlers or `LettaAgent` methods should be `async def` and
  use the existing fixtures rather than wrapping sync test harnesses around async code.
- **Don't add fallback/legacy branches for hypothetical old clients.** This codebase
  already carries one legacy path (`send_message_to_agent` vs. the `LettaAgent` loop)
  because of a real, documented compatibility need (agents not yet eligible for the
  new loop). Don't add a second one speculatively — extend eligibility criteria
  instead of forking the code path further.
- **Debug logging**: prefer the existing `debug_log(at_user_id, ...)` /
  `logger.warning("[TAG] ...")` conventions already used in `agents.py` (e.g.
  `[SEND_MESSAGE]`, `[TASK_LATENCY]`, `[CHAIN_OBS]`) over ad-hoc `print`/new logger
  patterns, so log lines stay greppable by existing dashboards and on-call runbooks.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **letta** (11747 symbols, 26393 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main-custom"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/letta/context` | Codebase overview, check index freshness |
| `gitnexus://repo/letta/clusters` | All functional areas |
| `gitnexus://repo/letta/processes` | All execution flows |
| `gitnexus://repo/letta/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
