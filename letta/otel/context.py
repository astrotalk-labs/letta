from contextvars import ContextVar
from typing import Any

# Create context var at module level (outside middleware)
request_attributes: ContextVar[dict[str, Any]] = ContextVar("request_attributes", default={})


# Helper functions
def set_ctx_attributes(attrs: dict[str, Any]):
    """Set attributes in current context"""
    current = request_attributes.get()
    new_attrs = {**current, **attrs}
    request_attributes.set(new_attrs)


def add_ctx_attribute(key: str, value: Any):
    """Add single attribute to current context"""
    current = request_attributes.get()
    new_attrs = {**current, key: value}
    request_attributes.set(new_attrs)


def get_ctx_attributes() -> dict[str, Any]:
    """Get all attributes from current context"""
    return request_attributes.get()
