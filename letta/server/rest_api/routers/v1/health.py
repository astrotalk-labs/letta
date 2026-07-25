
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from letta import __version__
from letta.log import get_logger
from letta.schemas.health import Health
from letta.server.db import get_db_async
from letta.server.utils import check_all_ai_services, check_database_health

logger = get_logger(__name__)



router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=Health, operation_id="health_check")
def health_check():
    return Health(
        version=__version__,
        status="ok",
    )


@router.get("/status", response_model=Health, operation_id="health_status_check")
async def health_status_check(db: AsyncSession = Depends(get_db_async)):
    db_health = await check_database_health(db)

    ai_services = await check_all_ai_services()

    all_healthy = db_health.healthy

    if ai_services:
        all_healthy = all_healthy and all(service.healthy for service in ai_services.values())

    if not all_healthy:
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
            },
        )

    return Health(
        version=__version__,
        status="ok",
    )
