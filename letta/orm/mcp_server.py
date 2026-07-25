
from sqlalchemy import JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from letta.functions.mcp_client.types import StdioServerConfig
from letta.orm.custom_columns import MCPStdioServerConfigColumn

# TODO everything in functions should live in this model
from letta.orm.enums import MCPServerType
from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.mcp import MCPServer


class MCPServer(SqlalchemyBase, OrganizationMixin):
    """Represents a registered MCP server"""

    __tablename__ = "mcp_server"
    __pydantic_model__ = MCPServer

    # Add unique constraint on (name, _organization_id)
    # An organization should not have multiple tools with the same name
    __table_args__ = (UniqueConstraint("server_name", "organization_id", name="uix_name_organization_mcp_server"),)

    server_name: Mapped[str] = mapped_column(doc="The display name of the MCP server")
    server_type: Mapped[MCPServerType] = mapped_column(
        String, default=MCPServerType.SSE, doc="The type of the MCP server. Only SSE is supported for remote servers."
    )

    # sse server
    server_url: Mapped[str | None] = mapped_column(
        String, nullable=True, doc="The URL of the server (MCP SSE client will connect to this URL)"
    )

    # access token / api key for MCP servers that require authentication
    token: Mapped[str | None] = mapped_column(String, nullable=True, doc="The access token or api key for the MCP server")

    # stdio server
    stdio_config: Mapped[StdioServerConfig | None] = mapped_column(
        MCPStdioServerConfigColumn, nullable=True, doc="The configuration for the stdio server"
    )

    metadata_: Mapped[dict | None] = mapped_column(
        JSON, default=dict, doc="A dictionary of additional metadata for the MCP server."
    )
    # relationships
    # organization: Mapped["Organization"] = relationship("Organization", back_populates="mcp_server", lazy="selectin")
