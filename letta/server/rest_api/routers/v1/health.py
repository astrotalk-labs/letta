from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from letta import __version__
from letta.log import get_logger
from letta.schemas.health import Health
from letta.server.db import get_db_async
from letta.server.utils import check_all_ai_services, check_database_health

logger = get_logger(__name__)


if TYPE_CHECKING:
    pass

router = APIRouter(prefix="/health", tags=["health"])


# Health check
@router.get("/", response_model=Health, operation_id="health_check")
async def health_check(db: AsyncSession = Depends(get_db_async)):
    # Check database health
    db_health = await check_database_health(db)
    
    # Check all configured AI services
    ai_services = await check_all_ai_services()
    
    # Determine overall health status
    all_healthy = db_health.healthy
    
    # Check if any AI service is unhealthy
    if ai_services:
        all_healthy = all_healthy and all(service.healthy for service in ai_services.values())
    
    # If any service is unhealthy, return 503
    if not all_healthy:
        # Log which services failed
        if not db_health.healthy:
            logger.error(f"Database unhealthy: {db_health.error}")
        for service_name, service_health in ai_services.items():
            if not service_health.healthy:
                logger.error(f"{service_name} unhealthy: {service_health.error}")
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "version": __version__,
                "status": "degraded",
            }
        )
    
    # Return original format
    return Health(
        version=__version__,
        status="ok",
    )

