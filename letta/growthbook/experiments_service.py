"""High-level experiments service – Python equivalent of AtExperimentsService.java.

Provides convenient methods for evaluating GrowthBook feature flags
with properly built attribute maps.
"""

from typing import Any, Dict, Optional, Type, TypeVar

from letta.growthbook.constants import GrowthBookAttributes
from letta.growthbook.feature_evaluator import FeatureEvaluator, FeatureEvaluatorException
from letta.log import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class ExperimentsService:
    """Convenience layer on top of :class:`FeatureEvaluator`.

    Builds attribute maps and evaluates experiments, mirroring the Java
    ``AtExperimentsService`` pattern.
    """

    def __init__(self, feature_evaluator: FeatureEvaluator):
        self._evaluator = feature_evaluator

    # ------------------------------------------------------------------
    # Attribute map builders
    # ------------------------------------------------------------------

    @staticmethod
    def build_attributes(
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        model_endpoint_type: Optional[str] = None,
        model: Optional[str] = None,
        extra_attributes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build an attributes dict for GrowthBook context.

        Only non-None values are included so GrowthBook targeting conditions
        that check for attribute existence work correctly.
        """
        attrs: Dict[str, Any] = {}
        if user_id is not None:
            attrs[GrowthBookAttributes.USER_ID] = user_id
        if agent_id is not None:
            attrs[GrowthBookAttributes.AGENT_ID] = agent_id
        if organization_id is not None:
            attrs[GrowthBookAttributes.ORGANIZATION_ID] = organization_id
        if model_endpoint_type is not None:
            attrs[GrowthBookAttributes.MODEL_ENDPOINT_TYPE] = model_endpoint_type
        if model is not None:
            attrs[GrowthBookAttributes.MODEL] = model
        if extra_attributes:
            attrs.update(extra_attributes)
        return attrs

    # ------------------------------------------------------------------
    # Generic evaluation helpers
    # ------------------------------------------------------------------

    def evaluate_feature(
        self,
        feature_key: str,
        attributes: Dict[str, Any],
        result_type: Type[T] = object,
    ) -> Optional[T]:
        """Evaluate a feature flag. Returns ``None`` on failure (fail-open)."""
        try:
            return self._evaluator.get_feature_value(feature_key, attributes, result_type)
        except FeatureEvaluatorException as e:
            logger.error(f"Error evaluating feature '{feature_key}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error evaluating feature '{feature_key}': {e}")
            return None

    def is_feature_on(self, feature_key: str, attributes: Dict[str, Any]) -> bool:
        """Boolean check -- ``True`` if the feature is on, ``False`` otherwise."""
        return self._evaluator.is_feature_on(feature_key, attributes)
