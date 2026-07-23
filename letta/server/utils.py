def condition_to_stop_receiving(response):
    """Determines when to stop listening to the server"""
    if response.get("type") in ["agent_response_end", "agent_response_error", "command_response", "server_error"]:
        return True
    else:
        return False


def print_server_response(response):
    """Turn response json into a nice print"""
    if response["type"] == "agent_response_start":
        print("[agent.step start]")
    elif response["type"] == "agent_response_end":
        print("[agent.step end]")
    elif response["type"] == "agent_response":
        msg = response["message"]
        if response["message_type"] == "internal_monologue":
            print(f"[inner thoughts] {msg}")
        elif response["message_type"] == "assistant_message":
            print(f"{msg}")
        elif response["message_type"] == "function_message":
            pass
        else:
            print(response)
    else:
        print(response)


def shorten_key_middle(key_string, chars_each_side=3):
    """
    Shortens a key string by showing a specified number of characters on each side and adding an ellipsis in the middle.

    Args:
    key_string (str): The key string to be shortened.
    chars_each_side (int): The number of characters to show on each side of the ellipsis.

    Returns:
    str: The shortened key string with an ellipsis in the middle.
    """
    if not key_string:
        return key_string
    key_length = len(key_string)
    if key_length <= 2 * chars_each_side:
        return "..."  # Return ellipsis if the key is too short
    else:
        return key_string[:chars_each_side] + "..." + key_string[-chars_each_side:]


# ============================================================================
# Health Check Utilities
# ============================================================================

import asyncio
import os
from typing import Dict, Optional

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from letta.log import get_logger
from letta.settings import model_settings

logger = get_logger(__name__)


class ServiceHealth(BaseModel):
    """Internal model for service health status"""

    service: str
    healthy: bool
    error: Optional[str] = None


async def check_database_health(db: AsyncSession) -> ServiceHealth:
    """
    Check database connectivity by performing a simple query.

    Args:
        db: AsyncSession instance

    Returns:
        ServiceHealth with database connectivity status
    """
    try:
        # Execute a simple query to verify database connectivity
        result = await db.execute(text("SELECT 1"))
        result.scalar()

        return ServiceHealth(service="postgres", healthy=True, error=None)
    except Exception as e:
        # Log error type without exposing sensitive connection details
        logger.error(f"Database health check failed: {type(e).__name__}")
        return ServiceHealth(service="postgres", healthy=False, error="Database connection failed")


async def check_anthropic_health() -> Optional[ServiceHealth]:
    """
    Check Anthropic direct API connectivity if API key is configured.

    Returns:
        ServiceHealth if API key is configured, None otherwise
    """
    api_key = model_settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        return None

    try:
        import anthropic

        # Create a minimal client and test connectivity
        client = anthropic.Anthropic(api_key=api_key, max_retries=1, timeout=5.0)

        # Try to use count_tokens endpoint as a lightweight connectivity check
        try:
            client.messages.count_tokens(model="claude-sonnet-4-5-20250929", messages=[{"role": "user", "content": "test"}])
            return ServiceHealth(service="anthropic_direct", healthy=True, error=None)
        except Exception as e:
            # Log error type without exposing API key details
            logger.error(f"Anthropic direct API health check failed: {type(e).__name__}")
            return ServiceHealth(service="anthropic_direct", healthy=False, error="Anthropic API unavailable")
    except ImportError:
        logger.warning("Anthropic library not installed")
        return ServiceHealth(service="anthropic_direct", healthy=False, error="Anthropic library not installed")
    except Exception as e:
        logger.error(f"Anthropic direct health check failed: {type(e).__name__}")
        return ServiceHealth(service="anthropic_direct", healthy=False, error="Anthropic API check failed")


async def check_all_ai_services() -> Dict[str, ServiceHealth]:
    """
    Check connectivity to configured AI services (Anthropic direct API only).

    Returns:
        Dictionary mapping service names to their health status
    """
    # TODO: Claude health check temporarily bypassed
    # Uncomment below to re-enable Anthropic health check:
    # anthropic_health = await check_anthropic_health()
    # ai_services = {}
    # if anthropic_health is not None:
    #     ai_services[anthropic_health.service] = anthropic_health
    # return ai_services

    return {}
