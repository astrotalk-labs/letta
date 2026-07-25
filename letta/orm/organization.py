from typing import TYPE_CHECKING, Union

from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.organization import Organization as PydanticOrganization

if TYPE_CHECKING:
    from letta.orm.agent import Agent
    from letta.orm.agent_passage import AgentPassage
    from letta.orm.block import Block
    from letta.orm.file import FileMetadata
    from letta.orm.group import Group
    from letta.orm.identity import Identity
    from letta.orm.llm_batch_item import LLMBatchItem
    from letta.orm.llm_batch_job import LLMBatchJob
    from letta.orm.message import Message
    from letta.orm.provider import Provider
    from letta.orm.sandbox_config import AgentEnvironmentVariable, SandboxConfig
    from letta.orm.sandbox_environment_variable import SandboxEnvironmentVariable
    from letta.orm.source import Source
    from letta.orm.source_passage import SourcePassage
    from letta.orm.tool import Tool
    from letta.orm.user import User


class Organization(SqlalchemyBase):
    """The highest level of the object tree. All Entities belong to one and only one Organization."""

    __tablename__ = "organizations"
    __pydantic_model__ = PydanticOrganization

    name: Mapped[str] = mapped_column(doc="The display name of the organization.")
    privileged_tools: Mapped[bool] = mapped_column(doc="Whether the organization has access to privileged tools.")

    # relationships
    users: Mapped[list["User"]] = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    tools: Mapped[list["Tool"]] = relationship("Tool", back_populates="organization", cascade="all, delete-orphan")
    # mcp_servers: Mapped[List["MCPServer"]] = relationship("MCPServer", back_populates="organization", cascade="all, delete-orphan")
    blocks: Mapped[list["Block"]] = relationship("Block", back_populates="organization", cascade="all, delete-orphan")
    sources: Mapped[list["Source"]] = relationship("Source", back_populates="organization", cascade="all, delete-orphan")
    files: Mapped[list["FileMetadata"]] = relationship("FileMetadata", back_populates="organization", cascade="all, delete-orphan")
    sandbox_configs: Mapped[list["SandboxConfig"]] = relationship(
        "SandboxConfig", back_populates="organization", cascade="all, delete-orphan"
    )
    sandbox_environment_variables: Mapped[list["SandboxEnvironmentVariable"]] = relationship(
        "SandboxEnvironmentVariable", back_populates="organization", cascade="all, delete-orphan"
    )
    agent_environment_variables: Mapped[list["AgentEnvironmentVariable"]] = relationship(
        "AgentEnvironmentVariable", back_populates="organization", cascade="all, delete-orphan"
    )

    # relationships
    agents: Mapped[list["Agent"]] = relationship("Agent", back_populates="organization", cascade="all, delete-orphan")
    messages: Mapped[list["Message"]] = relationship("Message", back_populates="organization", cascade="all, delete-orphan")
    source_passages: Mapped[list["SourcePassage"]] = relationship(
        "SourcePassage", back_populates="organization", cascade="all, delete-orphan"
    )
    agent_passages: Mapped[list["AgentPassage"]] = relationship("AgentPassage", back_populates="organization", cascade="all, delete-orphan")
    providers: Mapped[list["Provider"]] = relationship("Provider", back_populates="organization", cascade="all, delete-orphan")
    identities: Mapped[list["Identity"]] = relationship("Identity", back_populates="organization", cascade="all, delete-orphan")
    groups: Mapped[list["Group"]] = relationship("Group", back_populates="organization", cascade="all, delete-orphan")
    llm_batch_jobs: Mapped[list["LLMBatchJob"]] = relationship("LLMBatchJob", back_populates="organization", cascade="all, delete-orphan")
    llm_batch_items: Mapped[list["LLMBatchItem"]] = relationship(
        "LLMBatchItem", back_populates="organization", cascade="all, delete-orphan"
    )

    @property
    def passages(self) -> list[Union["SourcePassage", "AgentPassage"]]:
        """Convenience property to get all passages"""
        return self.source_passages + self.agent_passages
