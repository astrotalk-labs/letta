# DB Queries on `POST /v1/agents/{id}/messages`

Every database query executed during a single `send_message` request, in request-flow
order, with the generated SQL, a ready-to-run `EXPLAIN (ANALYZE, BUFFERS)` statement, and
the index it needs (with current status).

> Stack: SQLAlchemy async ORM over PostgreSQL (asyncpg) + pgvector.
> Entry point: `letta/server/rest_api/routers/v1/agents.py::send_message` → `LettaAgent.step()`.

---

## Request flow (where queries originate)

```
send_message (agents.py:665)
├─ user_manager.get_actor_or_default_async            → 1 SELECT (users)
├─ agent_manager.get_agent_by_id_async                → 1 SELECT (agents) + N selectin relationship loads
└─ LettaAgent.step() / _step()  ── loop over steps ──
   ├─ helpers._prepare_in_context_messages_no_persist → message reads + create_many
   ├─ _create_llm_request_data_async
   │   ├─ message_manager.size_async                  → COUNT(messages)
   │   ├─ passage_manager.agent_passage_size_async    → COUNT(agent_passages)
   │   └─ _rebuild_memory_async                        → block reads + file-agent reads + (cond.) system-msg UPDATE
   ├─ [tool exec] archival_memory_search              → pgvector similarity search   ⚠️ NO INDEX
   ├─ [tool exec] core_memory_append / _replace       → block read-modify-write
   ├─ message_manager.create_many_messages_async      → INSERT messages + reselect
   ├─ agent_manager.set_in_context_messages_async     → UPDATE agents + full relationship refresh
   ├─ step_manager.log_step_async                     → INSERT steps          (fire-and-forget candidate)
   ├─ telemetry_manager.create_provider_trace_async   → INSERT provider_traces (fire-and-forget candidate)
   └─ summarizer.summarize (conditional)              → block create/attach/update
```

---

## 1. `user_manager.get_actor_or_default_async`
Resolve the acting user from the `user_id` header.

```sql
SELECT * FROM users WHERE id = $1 LIMIT 1;
```
```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM users WHERE id = 'user-123' LIMIT 1;
```
**Index:** PK on `users.id` — ✅ exists.

---

## 2. `agent_manager.get_agent_by_id_async`  `agent_manager.py:1067`
Load the agent. **All relationships use `lazy="selectin"`**, so even with
`include_relationships=["multi_agent_group"]` the ORM fires a batch of follow-up SELECTs.

### 2.0 Base agent row
```sql
SELECT agents.* FROM agents
WHERE agents.id = $1 AND agents.organization_id = $2;
```
```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT agents.* FROM agents
WHERE agents.id = 'agent-123' AND agents.organization_id = 'org-1';
```
**Index:** PK on `agents.id` — ✅. `organization_id` filter is the access predicate; PK lookup dominates so fine.

### 2.1–2.N selectin relationship loads (each a separate query)
| Relationship | Query | Needed index | Status |
|---|---|---|---|
| tools | `… tools JOIN tools_agents ON tool_id WHERE tools_agents.agent_id=$1` | `tools_agents(agent_id)` | ⚠️ only PK `(agent_id,tool_id)` — leading col OK for this read |
| sources | `… sources JOIN sources_agents WHERE agent_id=$1` | `sources_agents(agent_id)` | ⚠️ only PK `(agent_id,source_id)` — leading col OK |
| core_memory (blocks) | `… block JOIN blocks_agents WHERE agent_id=$1` | `blocks_agents(agent_id, block_id)` | ⚠️ has `(block_label,agent_id)` — **agent_id not leading** |
| tags | `… agents_tags WHERE agent_id=$1` | `agents_tags(agent_id, tag)` | ✅ `ix_agents_tags_agent_id_tag` |
| identities | `… identities JOIN identities_agents WHERE agent_id=$1` | `identities_agents(agent_id)` | ⚠️ only PK `(identity_id,agent_id)` — **agent_id not leading** |
| groups / multi_agent_group | `… "group" WHERE manager_agent_id=$1` | `group(manager_agent_id)` | verify |
| tool_exec_env_vars | `agent_environment_variable WHERE agent_id=$1` | `(agent_id)` | verify |
| file_agents | `file_agent WHERE agent_id=$1` | `(agent_id)` | verify |

```sql
-- example: blocks selectin
EXPLAIN (ANALYZE, BUFFERS)
SELECT block.* FROM block
JOIN blocks_agents ON block.id = blocks_agents.block_id
WHERE blocks_agents.agent_id = 'agent-123';
```

> ⚠️ **N+1-by-relationship**: with `include_relationships=None` the ORM loads *all* optional
> relationships. The hot path only needs a subset. **Pass an explicit minimal
> `include_relationships=[...]`** to skip loads that aren't used in the loop.

---

## 3. `_prepare_in_context_messages_no_persist_async`  `helpers.py`
Loads the in-context message window and persists the new user message.

### 3.1 `message_manager.get_message_by_id_async` (message_manager.py:45)
```sql
SELECT * FROM messages WHERE id = $1 AND organization_id = $2 LIMIT 1;
```
```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM messages WHERE id = 'msg-1' AND organization_id = 'org-1' LIMIT 1;
```
**Index:** PK on `messages.id` — ✅. Consider `messages(organization_id, id)` for the 2-col filter (minor).

### 3.2 `message_manager.get_messages_by_ids_async` (message_manager.py:64)
```sql
SELECT * FROM messages
WHERE id = ANY($1) AND organization_id = $2;
```
```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM messages
WHERE id = ANY(ARRAY['msg-1','msg-2','msg-3']) AND organization_id = 'org-1';
```
**Index:** PK on `messages.id` handles the `ANY`. ✅

### 3.3 `message_manager.create_many_messages_async` (message_manager.py:127)
Bulk INSERT, then reselect ordered by `created_at`.
```sql
INSERT INTO messages (id, role, content, agent_id, organization_id, created_at, ...) VALUES (...), (...);
SELECT * FROM messages WHERE id = ANY($1) ORDER BY created_at;
```
**Index:** reselect uses PK; ORDER BY covered by `ix_messages_created_at (created_at, id)` — ✅.

---

## 4. `message_manager.size_async`  `message_manager.py:345`
Count messages for the agent (drives context-window math).
```sql
SELECT COUNT(1) FROM messages
WHERE organization_id = $1 AND agent_id = $2;
```
```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT COUNT(1) FROM messages WHERE organization_id = 'org-1' AND agent_id = 'agent-123';
```
**Index:** ✅ `ix_messages_org_agent (organization_id, agent_id)` exists and covers this.

---

## 5. `passage_manager.agent_passage_size_async`  `passage_manager.py:1031`
Count archival passages for the agent.
```sql
SELECT COUNT(1) FROM agent_passages
WHERE organization_id = $1 AND agent_id = $2;
```
```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT COUNT(1) FROM agent_passages WHERE organization_id = 'org-1' AND agent_id = 'agent-123';
```
**Index:** ✅ `ix_agent_passages_org_agent (organization_id, agent_id)` exists.

---

## 6. `_rebuild_memory_async`  `base_agent.py:83`
Rebuilds the system prompt from blocks. **No loop-based N+1** — blocks and file-agents are
batch-loaded with `IN`. Latency (observed up to 26s) comes from large block `value` payloads,
large system-message JSON diffs, and cache-invalidating rewrites — not query count.

### 6.1 Load blocks by IDs (`block_manager.get_all_blocks_by_ids_async`, block_manager.py:241)
```sql
SELECT block.* FROM block
WHERE block.id = ANY($1) AND block.organization_id = $2;
```
```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT block.* FROM block
WHERE block.id = ANY(ARRAY['block-1','block-2']) AND block.organization_id = 'org-1';
```
**Index:** PK on `block.id` — ✅. `block(organization_id, id)` would help the access predicate (minor).

### 6.2 Load file blocks (`files_agents_manager.py:148`)
```sql
SELECT files_agents.* FROM files_agents
WHERE file_name = ANY($1) AND organization_id = $2;
```
```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT files_agents.* FROM files_agents
WHERE file_name = ANY(ARRAY['a.pdf','b.txt']) AND organization_id = 'org-1';
```
**Index:** ⚠️ **MISSING** — add `files_agents(organization_id, file_name)`.

### 6.3 Conditional system-message UPDATE (`message_manager.update_message_by_id_async`, message_manager.py:269)
Only when memory changed. Read-modify-write (SELECT by PK, then UPDATE by PK).
```sql
SELECT * FROM messages WHERE id = $1 AND organization_id = $2;
UPDATE messages SET content = $1, updated_at = NOW(), updated_by = $2 WHERE id = $3;
```
**Index:** PK on `messages.id` — ✅.

---

## 7. `archival_memory_search` tool — pgvector similarity  ⚠️ CRITICAL
`sqlalchemy_base.py:375` (and the combined source+agent UNION in `agent_manager_helper.py:590`).
Cosine-distance ordering over the `embedding` column.

```sql
SELECT * FROM agent_passages
WHERE organization_id = $1 AND agent_id = $2
ORDER BY embedding <=> $3::vector
LIMIT 50;
```
```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM agent_passages
WHERE organization_id = 'org-1' AND agent_id = 'agent-123'
ORDER BY embedding <=> '[0.1,0.2,...]'::vector
LIMIT 50;
```
Combined query (UNION of `source_passages` + `agent_passages`) orders the same way.

**Index:** ⚠️ **MISSING — NO VECTOR INDEX on `agent_passages.embedding` or `source_passages.embedding`.**
Every archival search is a **sequential scan + full sort**, O(n) in passage count. Add HNSW:
```sql
CREATE INDEX CONCURRENTLY ix_agent_passages_embedding_hnsw
ON agent_passages USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX CONCURRENTLY ix_source_passages_embedding_hnsw
ON source_passages USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```
(Use `<=>` = cosine distance → `vector_cosine_ops`. HNSW preferred over IVFFlat for frequent updates.)
This is the single biggest DB win on the hot path.

---

## 8. `core_memory_append` / `core_memory_replace` → `block_manager.create_or_update_block_async`  `block_manager.py:43`
Read-modify-write on a block (optimistic version lock).
```sql
SELECT * FROM block WHERE id = $1 AND organization_id = $2;          -- read
UPDATE block SET value = $1, version = version + 1
WHERE id = $2 AND organization_id = $3 AND version = $4;             -- write
```
**Index:** PK on `block.id` — ✅. Cannot be fire-and-forget (RMW with version lock).

---

## 9. `agent_manager.set_in_context_messages_async`  `agent_manager.py:1526`
Updates `agents.message_ids`. Delegates to `update_agent_async`, which **re-runs the full
selectin relationship batch from §2** twice (read + post-update refresh).
```sql
UPDATE agents SET message_ids = $1::json, updated_at = NOW(), last_updated_by_id = $2 WHERE id = $3;
-- + full selectin relationship reload (tools/sources/blocks/identities/...)
```
**Index:** UPDATE by PK — ✅. The cost is the **relationship refresh**, gated by the same
join-table indexes flagged in §2 (blocks_agents / identities_agents / tools_agents / sources_agents).

---

## 10. `step_manager.log_step_async`  `step_manager.py:105`  — pure INSERT
```sql
INSERT INTO steps (id, organization_id, agent_id, model, completion_tokens, prompt_tokens, total_tokens, ...) VALUES (...);
```
**Index:** PK + FK targets (org/job/provider PKs) — ✅. **Fire-and-forget candidate** (no read before write).
Optional reporting indexes (not on hot path): `steps(organization_id, agent_id)`, `steps(trace_id)`.

---

## 11. `telemetry_manager.create_provider_trace_async`  `telemetry_manager.py:24`  — pure INSERT
```sql
INSERT INTO provider_traces (id, organization_id, request_json, response_json, step_id, created_at, ...) VALUES (...);
```
**Index:** ✅ `ix_step_id (step_id)` exists. **Fire-and-forget candidate** — this is observability
data and should never block the user response (it was 15s in one prod trace).

---

## 12. Summarizer (conditional, when buffer overflows)  `ephemeral_summary_agent.py`
- block lookup by label → `get_block_with_label_async` (§ below)
- block create (if missing) → INSERT block  *(pure insert)*
- block attach (if new) → INSERT blocks_agents + UPDATE agents *(RMW)*
- block value update → UPDATE block.value + version *(RMW)*

### `agent_manager.get_block_with_label_async`  `agent_manager.py:1881`
⚠️ Loads the agent + **all** core_memory blocks via selectin, then filters by label **in Python**.
```sql
SELECT block.* FROM block JOIN blocks_agents ON block.id = blocks_agents.block_id
WHERE blocks_agents.agent_id = $1;   -- then Python loops to find label
```
**Index:** wants `blocks_agents(agent_id, block_id)` — ⚠️ only `(block_label, agent_id)` exists.
**Optimization:** filter by label in SQL instead of Python (push `block_label = $2` into the WHERE).

---

## Index gap summary (actionable)

| # | Priority | Table | Recommended index | Status | Why |
|---|---|---|---|---|---|
| 1 | 🔴 **Critical** | `agent_passages` | `USING hnsw (embedding vector_cosine_ops)` | **MISSING** | archival_memory_search = seq-scan + full sort, O(n) |
| 2 | 🔴 **Critical** | `source_passages` | `USING hnsw (embedding vector_cosine_ops)` | **MISSING** | same, on the UNION search path |
| 3 | 🟠 High | `files_agents` | `(organization_id, file_name)` | **MISSING** | hit on every `_rebuild_memory_async` |
| 4 | 🟠 High | `blocks_agents` | `(agent_id, block_id)` | partial (`(block_label,agent_id)`) | agent_id not leading → relationship loads + RMW deletes |
| 5 | 🟡 Med | `identities_agents` | `(agent_id)` | PK `(identity_id,agent_id)` only | selectin load on every agent fetch |
| 6 | 🟡 Med | `tools_agents` | `(agent_id)` | PK `(agent_id,tool_id)` — leading OK | selectin load (PK leading col already serves) |
| 7 | 🟡 Med | `sources_agents` | `(agent_id, source_id)` | PK `(agent_id,source_id)` — leading OK | selectin load (PK already serves) |
| 8 | 🟢 Low | `messages` | `(organization_id, id)` | PK on id | minor — 2-col point lookups |
| 9 | 🟢 Low | `block` | `(organization_id, id)` | PK on id | minor — access predicate |

### Non-index wins (bigger than most indexes here)
- **Make `step` + `provider_trace` inserts fire-and-forget** — they were 13–15s in prod and block the response for zero user value.
- **Trim `get_agent_by_id_async` / `set_in_context_messages_async` relationship loads** — pass explicit minimal `include_relationships`; the selectin batch + post-update refresh is the recurring cost, not any single query.
- **Push the `block_label` filter into SQL** in `get_block_with_label_async` instead of Python-side filtering.

### Suggested first PR (highest ROI, lowest risk)
```sql
-- 1 & 2: the only O(n) → O(log n) changes on the path
CREATE INDEX CONCURRENTLY ix_agent_passages_embedding_hnsw
  ON agent_passages USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX CONCURRENTLY ix_source_passages_embedding_hnsw
  ON source_passages USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- 3 & 4: cheap B-tree adds, hit every request
CREATE INDEX CONCURRENTLY ix_files_agents_org_file_name ON files_agents (organization_id, file_name);
CREATE INDEX CONCURRENTLY ix_blocks_agents_agent_id ON blocks_agents (agent_id, block_id);
```

> Validate each with the `EXPLAIN (ANALYZE, BUFFERS)` statements above on a prod-sized table
> before and after. For the vector indexes, confirm the plan switches from `Seq Scan` + `Sort`
> to an `Index Scan using …_hnsw`.
