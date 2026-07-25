from enum import Enum

from pydantic import Field

from letta.schemas.letta_base import LettaBase


class IdentityType(str, Enum):
    """
    Enum to represent the type of the identity.
    """

    org = "org"
    user = "user"
    other = "other"


class IdentityPropertyType(str, Enum):
    """
    Enum to represent the type of the identity property.
    """

    string = "string"
    number = "number"
    boolean = "boolean"
    json = "json"


class IdentityBase(LettaBase):
    __id_prefix__ = "identity"


class IdentityProperty(LettaBase):
    """A property of an identity"""

    key: str = Field(..., description="The key of the property")
    value: str | int | float | bool | dict = Field(..., description="The value of the property")
    type: IdentityPropertyType = Field(..., description="The type of the property")


class Identity(IdentityBase):
    id: str = IdentityBase.generate_id_field()
    identifier_key: str = Field(..., description="External, user-generated identifier key of the identity.")
    name: str = Field(..., description="The name of the identity.")
    identity_type: IdentityType = Field(..., description="The type of the identity.")
    project_id: str | None = Field(None, description="The project id of the identity, if applicable.")
    agent_ids: list[str] = Field(..., description="The IDs of the agents associated with the identity.")
    block_ids: list[str] = Field(..., description="The IDs of the blocks associated with the identity.")
    organization_id: str | None = Field(None, description="The organization id of the user")
    properties: list[IdentityProperty] = Field(default_factory=list, description="List of properties associated with the identity")


class IdentityCreate(LettaBase):
    identifier_key: str = Field(..., description="External, user-generated identifier key of the identity.")
    name: str = Field(..., description="The name of the identity.")
    identity_type: IdentityType = Field(..., description="The type of the identity.")
    project_id: str | None = Field(None, description="The project id of the identity, if applicable.")
    agent_ids: list[str] | None = Field(None, description="The agent ids that are associated with the identity.")
    block_ids: list[str] | None = Field(None, description="The IDs of the blocks associated with the identity.")
    properties: list[IdentityProperty] | None = Field(None, description="List of properties associated with the identity.")


class IdentityUpsert(LettaBase):
    identifier_key: str = Field(..., description="External, user-generated identifier key of the identity.")
    name: str = Field(..., description="The name of the identity.")
    identity_type: IdentityType = Field(..., description="The type of the identity.")
    project_id: str | None = Field(None, description="The project id of the identity, if applicable.")
    agent_ids: list[str] | None = Field(None, description="The agent ids that are associated with the identity.")
    block_ids: list[str] | None = Field(None, description="The IDs of the blocks associated with the identity.")
    properties: list[IdentityProperty] | None = Field(None, description="List of properties associated with the identity.")


class IdentityUpdate(LettaBase):
    identifier_key: str | None = Field(None, description="External, user-generated identifier key of the identity.")
    name: str | None = Field(None, description="The name of the identity.")
    identity_type: IdentityType | None = Field(None, description="The type of the identity.")
    agent_ids: list[str] | None = Field(None, description="The agent ids that are associated with the identity.")
    block_ids: list[str] | None = Field(None, description="The IDs of the blocks associated with the identity.")
    properties: list[IdentityProperty] | None = Field(None, description="List of properties associated with the identity.")
