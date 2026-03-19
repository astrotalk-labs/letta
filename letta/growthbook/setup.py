"""GrowthBook singleton setup and lifecycle management.

Provides ``get_experiments_service()`` to obtain the global singleton,
and ``init_growthbook()`` / ``shutdown_growthbook()`` for app lifecycle hooks.
"""

from typing import Optional

from letta.growthbook.experiments_service import ExperimentsService
from letta.growthbook.feature_evaluator import FeatureEvaluator
from letta.growthbook.gb_features_service import GBFeaturesService
from letta.log import get_logger
from letta.settings import settings

logger = get_logger(__name__)

# Module-level singletons
_gb_features_service: Optional[GBFeaturesService] = None
_feature_evaluator: Optional[FeatureEvaluator] = None
_experiments_service: Optional[ExperimentsService] = None


def init_growthbook() -> Optional[ExperimentsService]:
    """Initialize the GrowthBook stack from application settings.

    Reads ``LETTA_GROWTHBOOK_CLIENT_KEY``, ``LETTA_GROWTHBOOK_API_HOST``,
    ``LETTA_GROWTHBOOK_FEATURES_JSON_PATH``, and ``LETTA_GROWTHBOOK_REFRESH_INTERVAL``
    from environment / settings.

    Safe to call multiple times – subsequent calls are no-ops.

    Returns:
        The :class:`ExperimentsService` singleton, or ``None`` if GrowthBook
        is not configured (both client_key and features_json_path are empty).
    """
    global _gb_features_service, _feature_evaluator, _experiments_service

    if _experiments_service is not None:
        return _experiments_service

    client_key = settings.growthbook_client_key
    api_host = settings.growthbook_api_host
    features_path = settings.growthbook_features_json_path
    refresh_interval = settings.growthbook_refresh_interval

    logger.info("Initializing GrowthBook...")
    _gb_features_service = GBFeaturesService(
        client_key=client_key,
        api_host=api_host,
        features_json_path=features_path,
        refresh_interval=refresh_interval,
    )
    _feature_evaluator = FeatureEvaluator(_gb_features_service)
    _experiments_service = ExperimentsService(_feature_evaluator)
    logger.info("GrowthBook initialized successfully.")
    return _experiments_service


def shutdown_growthbook():
    """Shut down the GrowthBook background refresh thread.

    Call this during application shutdown (e.g., FastAPI ``on_event("shutdown")``).
    """
    global _gb_features_service, _feature_evaluator, _experiments_service

    if _gb_features_service is not None:
        _gb_features_service.shutdown()
        logger.info("GrowthBook shut down.")

    _gb_features_service = None
    _feature_evaluator = None
    _experiments_service = None


def get_experiments_service() -> Optional[ExperimentsService]:
    """Return the global :class:`ExperimentsService` singleton.

    Returns ``None`` if GrowthBook was not initialized (not configured).
    This allows callers to gracefully fall back to defaults when GrowthBook
    is disabled.
    """
    return _experiments_service


def get_feature_evaluator() -> Optional[FeatureEvaluator]:
    """Return the global :class:`FeatureEvaluator` singleton."""
    return _feature_evaluator
