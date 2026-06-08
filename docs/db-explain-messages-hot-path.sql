-- ============================================================================
-- EXPLAIN queries for every DB query on POST /v1/agents/{id}/messages
-- ----------------------------------------------------------------------------
-- Replace the example literals (org-1, agent-123, msg-1, the ::vector array)
-- with real prod-sized values before running. Run each block against a
-- production-sized table. For the vector queries, confirm the plan switches
-- from `Seq Scan` + `Sort` to `Index Scan using ..._hnsw` once the index exists.
-- See docs/db-queries-messages-hot-path.md for context per query.
-- ============================================================================


-- 1. user_manager.get_actor_or_default_async --------------------------------
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM users WHERE id = 'user-123' LIMIT 1;


-- 2.0 agent_manager.get_agent_by_id_async — base row ------------------------
EXPLAIN (ANALYZE, BUFFERS)
SELECT agents.* FROM agents
WHERE agents.id = 'agent-123' AND agents.organization_id = 'org-1';

-- 2.1 selectin: tools
EXPLAIN (ANALYZE, BUFFERS)
SELECT tools.* FROM tools
JOIN tools_agents ON tools.id = tools_agents.tool_id
WHERE tools_agents.agent_id = 'agent-123';

-- 2.2 selectin: sources
EXPLAIN (ANALYZE, BUFFERS)
SELECT sources.* FROM sources
JOIN sources_agents ON sources.id = sources_agents.source_id
WHERE sources_agents.agent_id = 'agent-123';

-- 2.3 selectin: core_memory (blocks)
EXPLAIN (ANALYZE, BUFFERS)
SELECT block.* FROM block
JOIN blocks_agents ON block.id = blocks_agents.block_id
WHERE blocks_agents.agent_id = 'agent-123';

-- 2.4 selectin: tags
EXPLAIN (ANALYZE, BUFFERS)
SELECT agents_tags.* FROM agents_tags WHERE agents_tags.agent_id = 'agent-123';

-- 2.5 selectin: identities
EXPLAIN (ANALYZE, BUFFERS)
SELECT identities.* FROM identities
JOIN identities_agents ON identities.id = identities_agents.identity_id
WHERE identities_agents.agent_id = 'agent-123';

-- 2.6 selectin: multi_agent_group
EXPLAIN (ANALYZE, BUFFERS)
SELECT "group".* FROM "group" WHERE "group".manager_agent_id = 'agent-123';

-- 2.7 selectin: tool_exec_environment_variables
EXPLAIN (ANALYZE, BUFFERS)
SELECT agent_environment_variable.* FROM agent_environment_variable
WHERE agent_environment_variable.agent_id = 'agent-123';

-- 2.8 selectin: file_agents
EXPLAIN (ANALYZE, BUFFERS)
SELECT file_agent.* FROM file_agent WHERE file_agent.agent_id = 'agent-123';


-- 3.1 message_manager.get_message_by_id_async -------------------------------
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM messages WHERE id = 'msg-1' AND organization_id = 'org-1' LIMIT 1;

-- 3.2 message_manager.get_messages_by_ids_async
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM messages
WHERE id = ANY(ARRAY['msg-1','msg-2','msg-3']) AND organization_id = 'org-1';

-- 3.3 create_many_messages_async — post-insert reselect
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM messages
WHERE id = ANY(ARRAY['msg-1','msg-2']) ORDER BY created_at;


-- 4. message_manager.size_async ---------------------------------------------
EXPLAIN (ANALYZE, BUFFERS)
SELECT COUNT(1) FROM messages
WHERE organization_id = 'org-1' AND agent_id = 'agent-123';


-- 5. passage_manager.agent_passage_size_async -------------------------------
EXPLAIN (ANALYZE, BUFFERS)
SELECT COUNT(1) FROM agent_passages
WHERE organization_id = 'org-1' AND agent_id = 'agent-123';


-- 6.1 _rebuild_memory_async: load blocks by IDs -----------------------------
EXPLAIN (ANALYZE, BUFFERS)
SELECT block.* FROM block
WHERE block.id = ANY(ARRAY['block-1','block-2']) AND block.organization_id = 'org-1';

-- 6.2 _rebuild_memory_async: load file blocks   [needs files_agents(org,file_name)]
EXPLAIN (ANALYZE, BUFFERS)
SELECT files_agents.* FROM files_agents
WHERE file_name = ANY(ARRAY['a.pdf','b.txt']) AND organization_id = 'org-1';

-- 6.3 _rebuild_memory_async: read system message before UPDATE
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM messages WHERE id = 'msg-sys-1' AND organization_id = 'org-1';


-- 7. archival_memory_search — pgvector  [CRITICAL: needs HNSW index] --------
-- agent passages only:
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM agent_passages
WHERE organization_id = 'org-1' AND agent_id = 'agent-123'
ORDER BY embedding <=> '[0.1,0.2,0.3]'::vector
LIMIT 50;

-- combined source + agent passages (UNION search path):
EXPLAIN (ANALYZE, BUFFERS)
WITH combined_passages AS (
  SELECT id, text, embedding, created_at, agent_id, source_id, organization_id
  FROM source_passages
  WHERE organization_id = 'org-1'
    AND source_id IN (SELECT source_id FROM sources_agents WHERE agent_id = 'agent-123')
  UNION ALL
  SELECT id, text, embedding, created_at, agent_id, NULL, organization_id
  FROM agent_passages
  WHERE organization_id = 'org-1' AND agent_id = 'agent-123'
)
SELECT * FROM combined_passages
ORDER BY embedding <=> '[0.1,0.2,0.3]'::vector
LIMIT 50;


-- 8. core_memory_append / _replace — block read-modify-write ----------------
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM block WHERE id = 'block-1' AND organization_id = 'org-1';


-- 9. set_in_context_messages_async — agent UPDATE ---------------------------
-- (UPDATE by PK; the cost is the selectin relationship refresh — see §2 queries)
EXPLAIN (ANALYZE, BUFFERS)
UPDATE agents SET message_ids = '["msg-1","msg-2"]'::json, updated_at = NOW()
WHERE id = 'agent-123';


-- 12. summarizer: get_block_with_label_async --------------------------------
-- (current: loads ALL agent blocks, filters label in Python)
EXPLAIN (ANALYZE, BUFFERS)
SELECT block.* FROM block
JOIN blocks_agents ON block.id = blocks_agents.block_id
WHERE blocks_agents.agent_id = 'agent-123';

-- (proposed: push label filter into SQL)
EXPLAIN (ANALYZE, BUFFERS)
SELECT block.* FROM block
JOIN blocks_agents ON block.id = blocks_agents.block_id
WHERE blocks_agents.agent_id = 'agent-123'
  AND blocks_agents.block_label = 'conversation_summary';


-- ============================================================================
-- Note: pure INSERTs (step_manager.log_step_async §10,
-- telemetry_manager.create_provider_trace_async §11, block create §12) are
-- omitted — no read/index analysis needed (FK-target PKs already exist).
-- They are fire-and-forget candidates, not query-plan problems.
-- ============================================================================
