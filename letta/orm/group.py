import uuid

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.group import Group as PydanticGroup


class Group(SqlalchemyBase, OrganizationMixin):

    __tablename__ = "groups"
    __pydantic_model__ = PydanticGroup

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: f"group-{uuid.uuid4()}")
    description: Mapped[str] = mapped_column(nullable=False, doc="")
    manager_type: Mapped[str] = mapped_column(nullable=False, doc="")
    manager_agent_id: Mapped[str | None] = mapped_column(String, ForeignKey("agents.id", ondelete="RESTRICT"), nullable=True, doc="")
    termination_token: Mapped[str | None] = mapped_column(nullable=True, doc="")
    max_turns: Mapped[int | None] = mapped_column(nullable=True, doc="")
    sleeptime_agent_frequency: Mapped[int | None] = mapped_column(nullable=True, doc="")
    max_message_buffer_length: Mapped[int | None] = mapped_column(nullable=True, doc="")
    min_message_buffer_length: Mapped[int | None] = mapped_column(nullable=True, doc="")
    turns_counter: Mapped[int | None] = mapped_column(nullable=True, doc="")
    last_processed_message_id: Mapped[str | None] = mapped_column(nullable=True, doc="")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="groups")
    agent_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, doc="Ordered list of agent IDs in this group")
    agents: Mapped[list["Agent"]] = relationship(
        "Agent", secondary="groups_agents", lazy="selectin", passive_deletes=True, back_populates="groups"
    )
    shared_blocks: Mapped[list["Block"]] = relationship(
        "Block", secondary="groups_blocks", lazy="selectin", passive_deletes=True, back_populates="groups"
    )
    manager_agent: Mapped["Agent"] = relationship("Agent", lazy="joined", back_populates="multi_agent_group")
