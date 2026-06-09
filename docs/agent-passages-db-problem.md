# `agent_passages` — Database Scaling Problem & Remediation Options

**Audience:** Engineering leadership
**Status:** Investigation complete, decision required
**Owner:** Sarijan Sharma
**Date:** 2026-06-08

---

## 1. Executive summary

The `agent_passages` table — which stores the long-term "archival memory" embeddings that power our AI agents' recall — has
grown to **19 TB / 1.06 billion rows** and has **no vector index**. As a result, a single archival-memory search inside an agent turn can take **30+ seconds**, and it is currently the
**single largest contributor to user-facing latency** on the chat endpoint.

Two back-to-back production requests measured this week each took **66 seconds end-to-end**, of which **~33 seconds was one archival-memory search**. This is consistent and reproducible, not an outlier.

This is an **architecture/capacity problem, not a tuning problem**. It cannot be fixed by adding a normal index. It requires a planned migration. This document lays out the evidence, the root cause, and four remediation options with their cost, effort, risk, and expected impact, followed by a recommended phased plan.

**Headline numbers:**

| Metric | Value |
|---|---|
| Table size | **19 TB** (17 TB embeddings, 1.4 TB rows, 0.2 TB indexes) |
| Row count | **1.06 billion** |
| Embedding size | 4096 dimensions, float32 = **16 KB per row** |
| Vector index | **None** |
| Partitioning | **None** (single flat table) |
| Worst-case single archival search | **~33 seconds** |
| Share of a slow request consumed by archival search | **~50%** |

---

## 2. What the problem looks like in production

Two real requests captured on 2026-06-08, end-to-end latency broken down with per-phase timing logs:

| Phase | Request A (`319839131`) | Request B (`319321176`) |
|---|---|---|
| **`archival_memory_search` (one call)** | **31,935 ms** | **33,307 ms** |
| Cascade gate retries (separate issue) | ~18,000 ms | ~18,000 ms |
| All LLM calls combined | ~11,000 ms | ~7,000 ms |
| Memory rebuild + context write (DB) | ~100 ms | ~100 ms |
| **Total request** | **66,126 ms** | **66,396 ms** |

Two different agents, two different users, **the same ~33-second archival search dominating each request.** Both requests breached the 60-second threshold that triggers our upstream LLM fallback — i.e. this latency is already degrading the production experience.

> Note: the "memory rebuild" and "context write" DB operations were specifically suspected earlier and have now been **ruled out** — they consistently run in single-digit milliseconds. On these requests the problem is isolated to the archival-memory vector search.

### 2.1 Latency has more than one shape — quantify the split before over-investing

A third production request (`319966671`, 60s total) showed a **different** profile: the archival search was **fast (330 ms)**, but the agent took **14 sequential steps**, each a 3–7s LLM call over a large, growing prompt (prompt tokens climbed 12k → 34k; ~245k total), for ~56s of LLM time. The database was not the bottleneck on this request at all.

So slow requests fall into at least two buckets:

| Pattern | Dominant cost | Addressed by this document? |
|---|---|---|
| **DB-bound** (Requests A, B) | one ~33s archival-memory search | ✅ Yes — the plan below |
| **Step-count-bound** (Request C) | many LLM calls × large growing context | ❌ No — a separate agent-loop / context-size / prompt-cache problem |

**Recommendation:** before committing to the full migration (Phase 2 below), measure what fraction of slow requests are DB-bound vs step-count-bound, using the per-call latency metrics we now emit. The archival-search problem is real and worth fixing regardless (it is the *worst* single cost we see, and a cost/scale problem independent of latency), but sizing the split ensures the engineering investment targets the majority of the pain. The honest headline is: **archival search is the largest single latency cost we observe (up to ~33s) and a standalone storage/cost problem — but it is not the only driver of slow requests.**

---

## 3. Root cause

### 3.1 The table is 89% raw embedding vectors

Breakdown of the 19 TB:

| Component | Size | What it is |
|---|---|---|
| TOAST (out-of-line storage) | **17 TB** | the embedding vectors (4096-dim × 1.06B rows) |
| Heap (the rows) | 1.4 TB | text, ids, timestamps |
| Indexes | 199 GB | 4 standard B-tree indexes (id, org, org+agent, created_at) |

The embeddings *are* the table. Everything else is rounding error.

### 3.2 There is no vector index, so every search is a full scan

A vector search asks "find the 50 passages most similar to this query." With no vector index, PostgreSQL must:

1. Read **every** passage belonging to that agent from disk (each row is ~16 KB),
2. Compute the similarity score for all of them,
3. Sort, and return the top 50.

For an agent with a large archival memory, that means reading **multiple gigabytes from disk and sorting it, on the critical path of a live user request.** This is the 33 seconds.

The query plan confirms it: PostgreSQL performs a **sequential scan + full sort** (`Sort ... Sort Key: (embedding <=> $query)`), which is the textbook signature of a missing vector index.

### 3.3 Why a normal index can't be added

The standard fix — an approximate-nearest-neighbour index (HNSW) — **cannot be built globally on this table**:

- **Build cost:** building an HNSW index on 1.06 billion 4096-dimension vectors would take days, require enormous memory, and produce an index measured in **terabytes** — on top of the 19 TB we already have.
- **It wouldn't even work well:** HNSW indexes order results by *global* similarity across all 1B vectors. Our searches are always filtered to a single agent. The index would have to wade through millions of *other* agents' vectors before finding enough for the requesting agent — a known weakness called the "filtered-ANN problem." Slower and less accurate.

This is precisely why the fix is structural (partitioning), not a one-line index addition.

### 3.4 Unbounded growth

The table has **zero deleted/expired rows** — nothing is ever pruned. Growth is uniform across millions of agents (no single "abusive" agent to cap). At the current trajectory the table will keep growing past 19 TB indefinitely, compounding both the latency and the cost problem.

---

## 4. Why this matters (business impact)

1. **User-facing latency.** Archival search is ~50% of our slowest requests and pushes them past the 60-second fallback threshold. Every agent that relies on long-term memory recall is affected.
2. **Cost.** 19 TB of primary database storage is expensive on its own — and it is multiplied by every read replica and every backup. A 19 TB database is also slow and risky to back up, restore, and perform maintenance (`VACUUM`) on.
3. **Operational risk.** Large single tables make schema changes, failover, and disaster recovery slower and more fragile. The bigger it gets, the harder any future fix becomes.
4. **It gets worse with time.** With no retention, the problem compounds. Acting now is cheaper than acting at 40 TB.

---

## 5. Options to fix

Each option is rated on **Latency impact**, **Storage/cost impact**, **Effort**, and **Risk**.

### Option 0 — Do nothing (baseline)
- **Latency:** stays ~33s/search, worsening as the table grows.
- **Cost:** 19 TB and climbing.
- **Verdict:** Not viable — already breaching the latency fallback threshold in production.

---

### Option 1 — `halfvec` (store embeddings at half precision) — *storage + modest latency*
Convert the embedding column from 32-bit floats to 16-bit floats (a native, supported PostgreSQL/pgvector type).

- **How it works:** halves the bytes per vector (16 KB → 8 KB) with negligible loss of recall quality (16-bit floats retain ~3 significant digits, more than enough for similarity ranking).
- **Latency impact:** 🟡 Moderate. Searches are I/O-bound (reading those fat rows), so halving row size roughly **halves scan time** — but a 33s search becomes ~16s, still too slow. **This is not a complete fix on its own.**
- **Storage/cost impact:** 🟢 Large. Reclaims **~8.5 TB** (19 TB → ~10.5 TB). Directly cuts storage, backup, and replica costs.
- **Effort:** 🟡 Medium — a migration on a billion-row table (new column or new table, batched backfill, cutover).
- **Risk:** 🟢 Low — recall impact is validated with a quick before/after comparison (we can run this test in minutes, no migration required).

---

### Option 2 — Partition by agent + per-partition vector index — *the real latency fix*
Split the one giant table into many smaller partitions keyed by agent, and build a vector (HNSW) index on each partition.

- **How it works:** Every archival search already filters by a single agent. If the table is partitioned by agent, each search touches only that agent's small partition, and a vector index on that partition makes the search **logarithmic instead of linear** — i.e. sub-second regardless of how big the agent's memory is.
- **Latency impact:** 🟢 **Largest.** This is the only option that turns 33s into **well under a second**. It eliminates the dominant latency source entirely.
- **Storage/cost impact:** 🟡 Neutral-to-slightly-up (indexes add some space), unless combined with Option 1.
- **Effort:** 🔴 High — the biggest project here. Requires a new partitioned schema, a batched data migration of 1B rows, application/ORM changes, and a careful cutover.
- **Risk:** 🟡 Medium — large migration, needs staging validation and a dual-write or maintenance-window cutover plan. Well-understood pattern, but real engineering.

---

### Option 3 — Retention / TTL cap on archival passages — *stops the bleeding*
Introduce a policy limiting how many (or how old) archival passages each agent keeps.

- **How it works:** e.g. keep the most-recent or most-relevant N passages per agent; expire the rest. Because growth is uniform across agents, a per-agent cap shrinks the whole table proportionally.
- **Latency impact:** 🟡 Moderate — smaller per-agent sets mean faster scans even without an index.
- **Storage/cost impact:** 🟢 Large and ongoing — the only option that **stops unbounded growth**, not just a one-time reclaim.
- **Effort:** 🟡 Medium — needs a product decision on the policy, plus a background purge job.
- **Risk:** 🟡 Medium — it's a **product/quality decision** (we are choosing to forget old memories). Needs product sign-off, not just engineering.

---

### Option 4 — Lower the embedding dimension / dedicated vector store — *strategic, forward-looking*
We currently use a **4096-dimension** embedding — unusually large. Most retrieval quality saturates far below that; a 1024-dimension model would be **4× smaller and faster** with often-negligible quality loss. Alternatively, move vectors out of PostgreSQL into a purpose-built vector database.

- **How it works:** smaller vectors = proportionally smaller table and faster search; or offload the vector workload to a system designed for it.
- **Latency/cost impact:** 🟢 Potentially large on both.
- **Effort:** 🔴 Very high — lowering dimensions requires **re-embedding all 1B passages** (a large, expensive batch job); a dedicated store is a new piece of infrastructure to operate.
- **Risk:** 🔴 High — changes recall behaviour and/or adds operational surface. Best treated as a longer-term strategic track, with new data adopting the smaller dimension going forward.

---

## 6. Options at a glance

| Option | Latency fix | Storage saved | Stops growth | Effort | Risk |
|---|---|---|---|---|---|
| 0. Do nothing | ❌ worsens | — | ❌ | — | — |
| 1. `halfvec` | 🟡 ~2× | 🟢 ~8.5 TB | ❌ | 🟡 Med | 🟢 Low |
| 2. Partition + per-partition HNSW | 🟢 **33s → <1s** | 🟡 neutral | ❌ | 🔴 High | 🟡 Med |
| 3. Retention cap | 🟡 moderate | 🟢 ongoing | ✅ **yes** | 🟡 Med | 🟡 Med (product) |
| 4. Lower dims / vector store | 🟢 large | 🟢 large | partial | 🔴 V.High | 🔴 High |

---

## 7. Recommendation — phased plan

No single option is sufficient alone; the right answer combines a few in sequence, lowest-risk-first.

**Phase 1 — Quick relief (low risk, weeks)**
- **`halfvec` migration (Option 1).** Reclaims ~8.5 TB immediately (cuts storage/backup/replica cost ~45%) and roughly halves search latency as a side effect. De-risked by a recall-validation test we can run before committing.
- **Retention policy (Option 3).** Decide and ship a per-agent cap to stop unbounded growth. Requires a product conversation now so the policy is ready.

**Phase 2 — The real latency fix (higher effort, the main project)**
- **Partition `agent_passages` by agent + per-partition HNSW index (Option 2).** This is what actually turns 33s into sub-second. Do it *after* `halfvec` so we partition a 10 TB table, not a 19 TB one. Folding `halfvec` and partitioning into one migration avoids rewriting the data twice.

**Phase 3 — Strategic (optional, longer-term)**
- **Adopt a smaller embedding dimension for new data (Option 4)** and evaluate whether a dedicated vector store is warranted as scale continues.

**Separately (not a DB change, immediate win):** the same production traces show a second ~18-second cost from AI-routing retries that already has a code fix written and validated on a test user. Graduating that fix is a near-zero-risk change that removes ~18s from these requests independently of the database work. *(Tracked separately; mentioned here because it appears in the same traces.)*

---

## 8. What we need to proceed

1. **Decision:** approve the phased plan (or a subset).
2. **Product input:** the retention policy (how much archival memory per agent do we guarantee?).
3. **Engineering time:** Phase 1 is a few weeks; Phase 2 is the larger migration project requiring staging validation and a cutover window.

---

## Appendix A — How the data was gathered

- Table size / row count: PostgreSQL catalog (`pg_total_relation_size`, `pg_class.reltuples`) and storage breakdown (`pg_relation_size` / `pg_indexes_size` / TOAST).
- Distribution & deletion rate: statistical sampling (`TABLESAMPLE`) — full scans time out on a 1B-row table.
- Per-row anatomy: `pg_column_size()` + `vector_dims()` on a sampled row (4096 dims, 16 KB embedding).
- Latency: per-phase production timing logs (`[TASK_LATENCY]`, `[REBUILD_TIMING]`, `[CTXWIN_TIMING]`) added specifically for this investigation.
- Query plan: `EXPLAIN (ANALYZE, BUFFERS)` confirming sequential scan + sort (no vector index).

## Appendix B — Current indexes on `agent_passages`
- `agent_passages_pkey` — primary key on `id`
- `agent_passages_org_idx` — `(organization_id)`
- `ix_agent_passages_org_agent` — `(organization_id, agent_id)`
- `agent_passages_created_at_id_idx` — `(created_at, id)`
- **No vector/embedding index.**
