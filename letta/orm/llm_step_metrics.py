from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Double,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from letta.orm.base import Base


class LLMStepMetrics(Base):
    """Per-turn LLM usage metrics with per-step breakdown JSON.

    One row per agent turn (_step() invocation). Written asynchronously
    after each turn completes — never blocks the agent response.
    """

    __tablename__ = "llm_step_metrics"
    __table_args__ = (Index("ix_llm_step_metrics_created_at", "created_at", "at_user_id"),)

    # primary key
    turn_id: Mapped[str] = mapped_column(String, primary_key=True)

    # identifiers
    agent_id: Mapped[str | None] = mapped_column(String, nullable=True)
    at_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    consultant_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    chat_order_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # turn-level token totals
    total_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    cache_write_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    cache_read_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # derived cache metrics
    cache_efficiency: Mapped[float | None] = mapped_column(Double, nullable=True)
    cache_rotation_efficiency: Mapped[float | None] = mapped_column(Double, nullable=True)

    # character counts
    input_char_count: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    output_char_count: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # flags / routing
    cache_version: Mapped[str | None] = mapped_column(String, nullable=True)
    latency_optimisation_flow: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    endpoint_type: Mapped[str | None] = mapped_column(String, nullable=True)
    primary_model: Mapped[str | None] = mapped_column(String, nullable=True)

    # latency (ms)
    latency_ms: Mapped[float | None] = mapped_column(Double, nullable=True)

    # per-step breakdown (JSON array)
    steps_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # timestamps (IST-aware)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # catch-all metadata
    metadata_: Mapped[str | None] = mapped_column("metadata", Text, nullable=True)
