from __future__ import annotations

import asyncio
import logging
from typing import Any

from letta.orm.llm_step_metrics import LLMStepMetrics

logger = logging.getLogger(__name__)


async def _write(payload: dict[str, Any]) -> None:
    from letta.server.db import db_registry

    try:
        async with db_registry.async_session() as session:
            row = LLMStepMetrics(
                turn_id=payload["turn_id"],
                agent_id=payload.get("agent_id"),
                at_user_id=payload.get("at_user_id"),
                consultant_id=payload.get("consultant_id"),
                chat_order_id=payload.get("chat_order_id"),
                total_steps=payload.get("total_steps"),
                input_tokens=payload.get("input_tokens"),
                output_tokens=payload.get("output_tokens"),
                cache_write_tokens=payload.get("cache_write_tokens"),
                cache_read_tokens=payload.get("cache_read_tokens"),
                cache_efficiency=payload.get("cache_efficiency"),
                cache_rotation_efficiency=payload.get("cache_rotation_efficiency"),
                input_char_count=payload.get("input_char_count"),
                output_char_count=payload.get("output_char_count"),
                cache_version=payload.get("cache_version"),
                latency_optimisation_flow=payload.get("latency_optimisation_flow"),
                endpoint_type=payload.get("endpoint_type"),
                primary_model=payload.get("primary_model"),
                latency_ms=payload.get("latency_ms"),
                steps_json=payload.get("steps_json"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                metadata_=payload.get("metadata"),
            )
            session.add(row)
            await session.commit()
    except Exception as e:
        logger.warning("[STEP_METRICS] write failed: %s", e, exc_info=True)


def fire(payload: dict[str, Any]) -> None:
    """Fire-and-forget async write. Must be called from an async context."""
    task = asyncio.create_task(_write(payload))

    def _on_done(t: asyncio.Task) -> None:
        try:
            t.result()
        except Exception as e:
            logger.warning("[STEP_METRICS] task raised: %s", e, exc_info=True)

    task.add_done_callback(_on_done)
