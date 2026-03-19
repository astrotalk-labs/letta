"""Core feature evaluation service – Python equivalent of FeatureEvaluatorService.java.

Uses the GrowthBook Python SDK to evaluate feature flags with user/agent attributes.
"""

import json
from typing import Any, Dict, Optional, Type, TypeVar

from letta.log import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class FeatureEvaluatorException(Exception):
    """Raised when feature evaluation fails."""

    pass


class FeatureEvaluator:
    """Evaluates GrowthBook feature flags.

    Equivalent to the Java ``FeatureEvaluatorService`` – creates a GrowthBook
    context from user attributes + cached feature definitions, then calls
    ``evalFeature()``.
    """

    def __init__(self, gb_features_service):
        """
        Args:
            gb_features_service: A ``GBFeaturesService`` instance that provides
                the current features JSON.
        """
        self._gb_features_service = gb_features_service

    def get_feature_value(
        self,
        feature_key: str,
        attributes: Dict[str, Any],
        result_type: Type[T] = object,
    ) -> Optional[T]:
        """Evaluate a feature flag and return its value.

        Args:
            feature_key: The GrowthBook feature key (e.g. ``"use-vertex-experiment"``).
            attributes: Dict of targeting attributes (userId, agentId, etc.).
            result_type: Expected return type (used for logging/casting).

        Returns:
            The feature value, or ``None`` if the feature is off / unknown.

        Raises:
            FeatureEvaluatorException: If required parameters are missing.
        """
        try:
            from growthbook import GrowthBook  # type: ignore[import-untyped]
        except ImportError:
            raise FeatureEvaluatorException(
                "growthbook package is not installed. "
                "Install it with: pip install growthbook"
            )

        if not feature_key:
            raise FeatureEvaluatorException("Invalid feature key: feature_key is empty")
        if not attributes:
            raise FeatureEvaluatorException("User attributes not present")

        features_json = self._gb_features_service.get_features_json()

        try:
            features = json.loads(features_json) if isinstance(features_json, str) else features_json
        except json.JSONDecodeError as e:
            raise FeatureEvaluatorException(f"Failed to parse features JSON: {e}")

        gb = GrowthBook(
            attributes=attributes,
            features=features,
        )

        try:
            result = gb.eval_feature(feature_key)

            if result.source == "unknownFeature":
                logger.debug(f"GrowthBook feature '{feature_key}' is unknown (not defined).")
                return None

            if not result.on:
                logger.debug(f"GrowthBook feature '{feature_key}' is OFF.")
                return None

            value = result.value
            logger.debug(
                f"GrowthBook feature '{feature_key}' evaluated: "
                f"value={value}, source={result.source}"
            )
            return value

        except Exception as e:
            raise FeatureEvaluatorException(
                f"Exception evaluating feature '{feature_key}': {e}"
            ) from e
        finally:
            gb.destroy()

    def is_feature_on(self, feature_key: str, attributes: Dict[str, Any]) -> bool:
        """Check if a feature flag is enabled (boolean shorthand).

        Args:
            feature_key: The GrowthBook feature key.
            attributes: Dict of targeting attributes.

        Returns:
            ``True`` if the feature is on, ``False`` otherwise.
        """
        try:
            from growthbook import GrowthBook  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("growthbook package not installed – defaulting to False.")
            return False

        features_json = self._gb_features_service.get_features_json()
        try:
            features = json.loads(features_json) if isinstance(features_json, str) else features_json
        except json.JSONDecodeError:
            return False

        gb = GrowthBook(
            attributes=attributes,
            features=features,
        )
        try:
            return gb.is_on(feature_key)
        except Exception as e:
            logger.error(f"Error checking feature '{feature_key}': {e}")
            return False
        finally:
            gb.destroy()
