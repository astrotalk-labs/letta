# Letta v2 Carry-Forward Inventory (fork 0.8.4 → stock 0.16.8)

**Purpose:** Everything the AstroTalk fork changed on top of stock Letta, so you can rebuild the needed pieces on a fresh **0.16.8 + pgvector** instance — and consciously drop the rest.

**How to read the tags:**
- **RE-PORT** — custom, no stock equivalent → re-apply on 0.16.8.
- **NATIVE-IN-0.16.8** — stock now does this → use stock, don't re-port.
- **CLIENT-CHANGE** — config/contract change → adjust v2 config and/or your chat backend.
- **DROP** — experiment, diagnostic, or reverted → do not carry.

> Source of truth: `git diff 0.8.4 main-custom` (38 source files, ~3,716 insertions) + merged PRs #1–125.

---

## 0. TL;DR — what to actually carry forward

**Must RE-PORT (real product/infra features with no stock equivalent):**
1. **AWS Bedrock via boto3** `invoke_model` (if 0.16.8's native Bedrock still mishandles inference-profile ARNs)
2. **Geo/cohort API-key + Bedrock-ARN routing** (`_COHORT_TO_KEY_ENV`, `_COHORT_TO_BEDROCK_ARN`) — multi-tenant billing isolation
3. **GrowthBook** feature-flag subsystem (only if you keep the Azure-embeddings split)
4. **Custom cohort/business request fields** (`user_cohort`, `business_id`) + the **client must send them**
5. **Tool sorting for cache stability** (cheap, likely still needed)
6. **Bedrock/Haiku malformed tool-use normalization** (real provider quirk)
7. **Summarizer concurrency fix** (IntegrityError race) + **summary length truncation** (verify not native)
8. **DB-readiness health endpoint** (`/v1/health/status`)

**NATIVE-IN-0.16.8 (drop yours, use stock):**
- Prompt caching (v1 + v2 splitters, cache-obs) — **stock does it natively**
- Extended thinking / `output_config` — native reasoning config
- Vertex AI client — native
- Prometheus `/metrics` — verify native, likely present
- Redis auth settings

**DROP (experiments / diagnostics / reverted):**
- **Haiku cascade** (latencyOptimisationFlow) — caused the runaway; re-measure latency on v2 first
- **v2 cache rollout** — you already disabled it (+17% cost regression)
- mid-turn & trivial memory-rebuild skips — rolled back
- all cache-observability + `[TASK_LATENCY]`/`[STEP_TIMING]`/`[SUMMARIZER_*]` diagnostic logging
- `task_id` latency plumbing

**CLIENT-CHANGE (your chat backend must change for v2):**
- `model_override` → **renamed** `override_model`
- `thinking` (dict) → `enable_thinking` (string)
- stop sending `use_bedrock_experiment` / `use_vertex_experiment` / `latencyOptimisationFlow` / `output_config` / `task_id`
- provider + cohort-key selection now via agent config, not request flags

---

## 1. PR history (#1–125) by theme

| Theme | PRs | Outcome |
|---|---|---|
| Prompt caching (system-prompt split, v1/v2, sort tools, cache-obs, skip-rebuild) | #3, #45–68 | Heavy effort; **v2 reverted to 0% (+17% cost)**, several skips rolled back |
| Azure embeddings + rollout | #21–43 | Azure `text-embedding-3-small` + GrowthBook-gated rollout |
| GrowthBook feature flags | #22 | New subsystem; only `use_azure_embeddings` consumed |
| Bedrock / multi-provider / Sonnet 4.6 | #35–36, #87–91 | boto3 Bedrock, model routing |
| Geo-key / cohort routing | #71–86 | Per-cohort API keys + Bedrock ARNs |
| Summarizer (GPT/Azure→Haiku) | #37–40, #60, #64 | Anthropic Haiku summarizer + caching |
| Archival-memory timeout | #69–70 | (no timeout actually present in final code) |
| Haiku cascade / latency | #93–116 | **Reverted twice** — runaway loop bug |
| Observability / metrics / OTel | #19–20, #50–59, #94–98, #103–104, #117, #121–125 | Logging, custom metrics, Prometheus, e2e latency |
| Bug fixes | #49, #52–53, #105–106 | mistralai, system-msg persist, tool-input parsing |

---

## 2. Carry-forward inventory by subsystem

### 2A. Agent loop & cascade (`letta/agents/`)

| # | Item | Key symbols | Tag |
|---|---|---|---|
| 1 | **Haiku cascade** (route tool-steps to Haiku, send_message on primary) — incl. rollout gate, tool restriction, 8s timeout, quality-gate retry, model-restore | `latency_optimisation_flow`, `_cascade_enabled_for_user`, `_CASCADE_ROLLOUT_PCT`, `_HAIKU_MODEL_BY_PROVIDER`, `_PRIMARY_RESERVED_TOOLS`, `_HAIKU_TIMEOUT_SECONDS` | **DROP** (re-measure latency on v2 first; it caused the runaway and the real bottleneck was archival memory) |
| 2 | **Provider switching** (runtime endpoint/model swap) | `_apply_provider_switching`, `use_vertex_experiment`, `use_bedrock_experiment`, `model_override` | **RE-PORT** if you still need per-request provider swap; else **NATIVE** (configure provider on the agent) |
| 3 | Latency/timing logging | `[TASK_LATENCY]`, `[STEP_TIMING]`, `[REBUILD_TIMING]`, `[CTXWIN_TIMING]`, `self._task_id` | **DROP** (use native OTel spans) |
| 4 | Custom cascade OTel metrics | `haiku_cascade_*`, `llm_call_ms_histogram` | **DROP** with cascade (keep `llm_call_ms` only if useful) |
| 5 | `thinking`/`output_config` passthrough | `_THINKING_GATED_USER_IDS` | **NATIVE-IN-0.16.8** (native reasoning config) |
| 6 | Mid-turn memory-rebuild skip | `skip_rebuild`, `[REBUILD_SKIPPED_MID_TURN]` | **DROP** (rolled back, +2.9% cost) |
| 7 | Trivial memory-rebuild skip | `SKIP_TRIVIAL_REBUILD_*`, `[MEMORY_REBUILD_SKIPPED]` | **DROP** (test-user-only, heuristic) |
| 8 | **Synthetic send_message fallback** (text→tool_call) | `synthetic_…` ToolCall wrap | **RE-PORT** (verify 0.16.8 doesn't already handle no-tool-call) |
| 9 | Streaming interface accepts vertex/bedrock endpoint types | `model_endpoint_type in (...)` | **RE-PORT** (only with #2) |
| 10 | Summarizer enablement keyed off Anthropic/Azure key (not OpenAI) | `enable_summarization` gate | **CLIENT-CHANGE** (set provider keys in config) |
| 11 | **Skip persisting system-role input messages** (per-turn system events not saved) | `_persistable_initial` filter | **RE-PORT** if your `go-ai-chat` client still injects per-turn system messages |

### 2B. LLM client, providers, caching & geo-key routing (`letta/llm_api/`)

| # | Item | Key symbols | Tag |
|---|---|---|---|
| 1 | **Bedrock via boto3 `invoke_model`** (ARN profiles, `bedrock-2023-05-31`) | bedrock branch in `request_async`, `_invoke()` | **RE-PORT** (only if 0.16.8 native Bedrock still mishandles ARN profiles — test first) |
| 2 | **Bedrock inference-profile ARN ladder** (verbatim prefixes → cohort → env-var → raw) | `_BEDROCK_DIRECT_PREFIXES`, `BEDROCK_*_INFERENCE_PROFILE_ARN` | **RE-PORT** (with #1) |
| 3 | Bedrock thinking-field adaptation (adaptive→enabled, drop unsupported) | `_supports_thinking`, `is_bedrock` | **CLIENT-CHANGE** (fold into 0.16.8 Bedrock builder if needed) |
| 4 | Vertex AI client (+ `global` region/base_url fix) | `AnthropicVertexClient`, base_url override | **NATIVE-IN-0.16.8** (keep only the `global` URL fix if still broken) |
| 5 | **Geo/cohort Anthropic API-key routing** | `_COHORT_TO_KEY_ENV`, `_resolve_key_from_cohort`, `GEO_KEY` | **RE-PORT** (multi-tenant billing; no stock equivalent) |
| 6 | **Geo/cohort Bedrock-ARN routing** | `_COHORT_TO_BEDROCK_ARN`, `_build_bedrock_arn` | **RE-PORT** (with #5) |
| 7 | Ephemeral caching — v1 splitter (4-part) | `_split_system_message_for_caching` | **NATIVE-IN-0.16.8** |
| 8 | Ephemeral caching — v2 splitter + rollout | `V2_CACHE_ROLLOUT_*`, `_split_system_message_v2_for_caching` | **DROP** (disabled at 0%, +17% cost regression) |
| 9 | Cache observability logging | `CACHE_OBS_*`, `_log_cache_observation_*` | **DROP** (diagnostic) |
| 10 | **Tool sorting for cache stability** | `sorted(tools_for_request, key=name)` | **RE-PORT** (cheap; verify 0.16.8 ORM ordering is deterministic) |
| 11 | Model-name `@`↔`-` conversion (Vertex vs direct) | `rsplit('-',1)`, `replace('@','-')` | **RE-PORT** only with provider switching; else **DROP** |
| 12 | Legacy sync provider-switching plumbing (`llm_api_tools.py`) | `ensure_correct_provider` | **DROP** (duplicate of async path; debug-print littered) |
| 13 | Vertex Claude models in provider listing | `claude_models` dict | **NATIVE-IN-0.16.8** (verify; else re-port small dict) |
| 14 | **Bedrock/Haiku malformed tool-use normalization** (JSON-string `input`/`function`/`arguments`) | `_is_nested_openai_format` | **RE-PORT** (real Bedrock quirk; keep if Bedrock path retained) |
| 15 | Empty-messages prefill guard + 4.6 prefill block | `prefill_blocked` | **CLIENT-CHANGE** (4.6 prefill block is a real API constraint) |

> Note: `anthropic.py` + `llm_api_tools.py` are full of `print("DEBUG: …")` — **DROP all debug prints** on re-port.

### 2C. Feature flags (GrowthBook), API request contract & settings

**GrowthBook** (`letta/growthbook/`, 6 new files): `GBFeaturesService` (polls feature defs, file fallback), `FeatureEvaluator` (wraps `growthbook` SDK), `ExperimentsService` (attribute builder), lifecycle singletons. Wired in `app.py` lifespan; **only `use_azure_embeddings` is actually consumed** (in `passage_manager` + `sources`). → **RE-PORT only if you keep the Azure-embeddings split; otherwise DROP the whole subsystem.** Move the hardcoded `growthbook_client_key` default to env-only.

**Custom request fields** (`letta/schemas/letta_request.py`, `populate_by_name=True`):

| Field | Tag | Note |
|---|---|---|
| `use_vertex_experiment` / `use_bedrock_experiment` | **RE-PORT**/CLIENT | only with provider switching; client stops sending if dropped |
| `model_override` | **CLIENT-CHANGE** | stock has `override_model` — **rename in client** |
| `user_cohort` | **RE-PORT** | only with cohort key routing |
| `business_id` | **RE-PORT**/DROP | AstroTalk domain field |
| `thinking` (dict) | **NATIVE** | stock `enable_thinking` (string) — **client changes shape** |
| `output_config` (dict) | **NATIVE** | reconcile with native effort/reasoning |
| `task_id` | **DROP** | observability-only |
| `latencyOptimisationFlow` | **DROP** | cascade flag |

**Settings** (`letta/settings.py`): Azure embedding keys (`embeddings_3_small_*`, `letta_embedding_5_4_mini_*`) — **RE-PORT/CLIENT** if Azure kept; Redis auth — **NATIVE**; Prometheus knobs (`otel_metrics_prometheus_*`) — **RE-PORT/NATIVE**; GrowthBook knobs — **RE-PORT with GrowthBook** (key → env-only).

### 2D. Observability, metrics & REST endpoints

| # | Item | Tag |
|---|---|---|
| 1 | Custom metrics: `hist_messages_endpoint_e2e_ms`, `hist_llm_call_ms`, 9× `haiku_cascade_*` | **RE-PORT** the e2e + llm_call ones; **DROP** cascade ones. **Must register explicit-bucket Views** (advisory boundaries are ignored → 10s cap). |
| 2 | Prometheus `/metrics` standalone server (dedicated port) | **NATIVE-IN-0.16.8** (verify; else RE-PORT) |
| 3 | OTel log exporting (`otel/logging.py` → Loki) | **RE-PORT** if no native log pipeline |
| 4 | Unhandled-exception traceback dump | **DROP** (noisy `print`; keep `logger.error(exc_info=)` only) |
| 5 | **DB-readiness health endpoint** `/v1/health/status` | **RE-PORT** (drop the dead `check_anthropic_health`) |
| 6 | Health-path password-middleware allowlist | **RE-PORT** (with #5) |
| 7 | Per-env service name + GrowthBook lifecycle in `app.py` | **RE-PORT** service-name (trivial); GrowthBook with 2C |
| 8 | `/messages` e2e latency recording + custom-field threading + `x_gb_user_id` header | **RE-PORT** (e2e) + **CLIENT-CHANGE** (field threading must match 0.16.8 `step()` signature); **DROP** `[SEND_MESSAGE]`/`[TASK_LATENCY]` warn-logs |

> **Single most important observability carry-forward:** register the **explicit-bucket Views** for any latency histogram (the 10k-cap bug). Don't ship advisory boundaries without Views.

### 2E. Summarizer, archival memory & embeddings

| # | Item | Tag |
|---|---|---|
| 1 | Summarizer swapped OpenAI→Anthropic Haiku (dead Azure branch) | **CLIENT-CHANGE** (configure summarizer provider natively; drop dead Azure branch) |
| 2 | Ephemeral `cache_control` on summarizer system prompt | **CLIENT-CHANGE** (stale `2024-07-31` beta; re-express via current client / partly NATIVE) |
| 3 | `[SUMMARIZER_FIRED]` log | **DROP** |
| 4 | `[SUMMARIZER_USAGE]` sampled cache log | **DROP** |
| 5 | **IntegrityError race fix** on conversation_summary attach | **RE-PORT** (verify not native) |
| 6 | **Summary length truncation** to block limit | **RE-PORT** (cheap guard) |
| 7 | Generalized embedding helper w/ Azure (`get_embedding`) | **CLIENT-CHANGE** (native Azure embedding provider) |
| 8 | GrowthBook + `uid%10<6` Azure embeddings 60% rollout | **DROP** (set endpoint via config once chosen) |
| 9 | `[Embeddings] Successfully stored` log | **DROP** |

**Stock-inherited but migration-critical (present in 0.16.8 too — NOT fork changes):**
| # | Item | Tag |
|---|---|---|
| 10 | **4096 zero-padding** of every embedding (`MAX_EMBEDDING_DIM=4096`, `pad_embeddings`, `Vector(4096)`) — the 19 TB bloat driver | **NATIVE (still present)** → **fix on v2: set `MAX_EMBEDDING_DIM=1024`** before any data |
| 11 | **No pgvector ANN index** on embedding (btrees only) | **NATIVE (still none)** → **add HNSW on the empty table** (needs dim ≤2000, i.e. fix #10 first) |
| 12 | Per-`agent_id`/`archive_id` keying + cosine_distance vector query | **NATIVE** → preserve keying; in 0.16.8 archival memory is keyed by **`archive_id`** |

---

## 3. v2 day-1 setup decisions (decided this investigation)

| Decision | Value | Why |
|---|---|---|
| **Embedding model + dim** | `text-embedding-3-small`, **`dimensions=1024`** | recall@10 = **0.955** vs 1536 (validated); ≤2000 → indexable; 4× less storage than 4096 |
| **`MAX_EMBEDDING_DIM`** | **1024** (set before any data; fresh DB, no reset cost) | removes the zero-padding that caused 19 TB; makes HNSW possible |
| **Vector index** | **HNSW on the empty `archival_passages.embedding`** from day 1 | avoids the multi-day retrofit; turns 33 s search → sub-second |
| **pgvector** | **≥ 0.8**, `hnsw.iterative_scan = relaxed_order` | needed so the `archive_id` filter still returns full top-k |
| **Retention** | per-archive count/time cap + purge job, **from day 1** | the root cause of unbounded 19 TB growth |
| **Bedrock** | **global** inference profile (not regional) | regional ap-south-1 degradation caused 24–33 s spikes |

> Validate 1024 once more at `LIMIT 100` on 2–3 agents before locking. Fallback: 1536 (still indexable, still better than 4096).

---

## 4. Migration notes (old → v2)

- **`.af` agent-file carries:** agent config, **core-memory blocks, message history, tools, files, groups**. **Does NOT carry archival passages.**
- **Archival memory = separate custom migration** into the new **Archive** model: `create_archive_async` → `attach_agent_to_archive_async` → `create_agent_passages_async` (each needs `archive_id`).
- **Embedding conversion:** truncate old 1536→1024 + renormalize (Matryoshka; no re-embedding cost). Apply **retention during the move** (don't carry the 19 TB).
- ⚠️ **Test `.af` 0.8.4 → 0.16.8 on one agent first** — the format evolved 8 versions; if it fails, do a custom DB→DB migration for everything.
- Track an **old→new ID map** (agent → agent → archive), idempotent for re-runs.

---

## 5. Prioritized action list

1. **Stand up v2** with `MAX_EMBEDDING_DIM=1024`, model `dimensions=1024`, HNSW index on empty table, retention job, global Bedrock profile, pgvector ≥0.8 + iterative scan.
2. **Update the chat backend** for the new contract: `model_override`→`override_model`, `thinking`→`enable_thinking`, stop sending experiment/cascade/task_id fields, move provider+cohort selection to agent config.
3. **RE-PORT the keepers:** geo/cohort key+ARN routing, Bedrock boto3 shim (if needed), tool sorting, Bedrock/Haiku tool-input normalization, summarizer race fix + truncation, DB-readiness health endpoint, e2e/llm_call metrics **with Views**, OTel log pipeline. GrowthBook + Azure embeddings only if you keep that split.
4. **Adopt native, drop yours:** caching, extended thinking, Vertex client, Prometheus (verify).
5. **DROP:** cascade, v2 cache rollout, all rebuild-skips, all diagnostic logging.
6. **Re-measure on v2** before re-adding any latency optimization — the real bottleneck was archival memory, now fixed by dim+index+retention.
