"""Unit tests for the GrowthBook infrastructure."""

import json
from unittest.mock import MagicMock, patch

import pytest

from letta.growthbook.constants import GrowthBookAttributes
from letta.growthbook.experiments_service import ExperimentsService
from letta.growthbook.feature_evaluator import FeatureEvaluator, FeatureEvaluatorException
from letta.growthbook.gb_features_service import GBFeaturesService


# ---------------------------------------------------------------------------
# GBFeaturesService
# ---------------------------------------------------------------------------


class TestGBFeaturesService:

    def test_init_empty(self):
        """No client_key and no file -- starts with empty features."""
        svc = GBFeaturesService()
        assert svc.get_features_json() == "{}"
        svc.shutdown()

    def test_init_from_file(self, tmp_path):
        """Loads features from a local JSON file."""
        features = {"my-flag": {"defaultValue": True}}
        f = tmp_path / "features.json"
        f.write_text(json.dumps(features))

        svc = GBFeaturesService(features_json_path=str(f))
        loaded = json.loads(svc.get_features_json())
        assert "my-flag" in loaded
        svc.shutdown()

    def test_init_from_missing_file(self, tmp_path):
        """Does not crash if the file doesn't exist."""
        svc = GBFeaturesService(features_json_path=str(tmp_path / "nonexistent.json"))
        assert svc.get_features_json() == "{}"
        svc.shutdown()


# ---------------------------------------------------------------------------
# FeatureEvaluator
# ---------------------------------------------------------------------------


class TestFeatureEvaluator:

    def _make_evaluator(self, features: dict) -> FeatureEvaluator:
        gb_svc = MagicMock(spec=GBFeaturesService)
        gb_svc.get_features_json.return_value = json.dumps(features)
        return FeatureEvaluator(gb_svc)

    def test_feature_on(self):
        evaluator = self._make_evaluator({"my-flag": {"defaultValue": True}})
        result = evaluator.get_feature_value("my-flag", {"userId": "123"})
        assert result is True

    def test_feature_off_returns_none(self):
        evaluator = self._make_evaluator({"my-flag": {"defaultValue": False}})
        result = evaluator.get_feature_value("my-flag", {"userId": "123"})
        assert result is None

    def test_unknown_feature_returns_none(self):
        evaluator = self._make_evaluator({})
        result = evaluator.get_feature_value("nonexistent", {"userId": "123"})
        assert result is None

    def test_is_feature_on_true(self):
        evaluator = self._make_evaluator({"my-flag": {"defaultValue": True}})
        assert evaluator.is_feature_on("my-flag", {"userId": "123"}) is True

    def test_is_feature_on_false(self):
        evaluator = self._make_evaluator({"my-flag": {"defaultValue": False}})
        assert evaluator.is_feature_on("my-flag", {"userId": "123"}) is False

    def test_empty_feature_key_raises(self):
        evaluator = self._make_evaluator({})
        with pytest.raises(FeatureEvaluatorException, match="Invalid feature key"):
            evaluator.get_feature_value("", {"userId": "123"})

    def test_empty_attributes_raises(self):
        evaluator = self._make_evaluator({})
        with pytest.raises(FeatureEvaluatorException, match="attributes not present"):
            evaluator.get_feature_value("my-flag", {})

    def test_json_value(self):
        evaluator = self._make_evaluator(
            {
                "config-flag": {"defaultValue": {"tier": "premium", "limit": 100}},
            }
        )
        result = evaluator.get_feature_value("config-flag", {"userId": "123"})
        assert result == {"tier": "premium", "limit": 100}


# ---------------------------------------------------------------------------
# ExperimentsService
# ---------------------------------------------------------------------------


class TestExperimentsService:

    def _make_service(self, features: dict) -> ExperimentsService:
        gb_svc = MagicMock(spec=GBFeaturesService)
        gb_svc.get_features_json.return_value = json.dumps(features)
        evaluator = FeatureEvaluator(gb_svc)
        return ExperimentsService(evaluator)

    def test_build_attributes(self):
        attrs = ExperimentsService.build_attributes(
            user_id="u1",
            agent_id="a1",
            organization_id="org1",
            extra_attributes={"custom": "val"},
        )
        assert attrs[GrowthBookAttributes.USER_ID] == "u1"
        assert attrs[GrowthBookAttributes.AGENT_ID] == "a1"
        assert attrs["custom"] == "val"

    def test_build_attributes_skips_none(self):
        attrs = ExperimentsService.build_attributes(user_id="u1")
        assert GrowthBookAttributes.AGENT_ID not in attrs

    def test_evaluate_feature(self):
        svc = self._make_service({"my-flag": {"defaultValue": True}})
        assert svc.evaluate_feature("my-flag", {"userId": "1"}) is True

    def test_evaluate_feature_unknown_returns_none(self):
        svc = self._make_service({})
        assert svc.evaluate_feature("nonexistent", {"userId": "1"}) is None

    def test_is_feature_on(self):
        svc = self._make_service({"my-flag": {"defaultValue": True}})
        assert svc.is_feature_on("my-flag", {"userId": "1"}) is True


# ---------------------------------------------------------------------------
# Setup / lifecycle
# ---------------------------------------------------------------------------


class TestGrowthBookSetup:

    def test_init_without_config_returns_none(self):
        from letta.growthbook.setup import init_growthbook, shutdown_growthbook

        shutdown_growthbook()

        with patch("letta.growthbook.setup.settings") as mock_settings:
            mock_settings.growthbook_client_key = None
            mock_settings.growthbook_features_json_path = None
            mock_settings.growthbook_api_host = "https://cdn.growthbook.io"
            mock_settings.growthbook_refresh_interval = 60

            result = init_growthbook()
            assert result is None

        shutdown_growthbook()

    def test_init_with_local_file(self, tmp_path):
        from letta.growthbook.setup import init_growthbook, shutdown_growthbook

        shutdown_growthbook()

        features = {"test-flag": {"defaultValue": True}}
        f = tmp_path / "features.json"
        f.write_text(json.dumps(features))

        with patch("letta.growthbook.setup.settings") as mock_settings:
            mock_settings.growthbook_client_key = None
            mock_settings.growthbook_features_json_path = str(f)
            mock_settings.growthbook_api_host = "https://cdn.growthbook.io"
            mock_settings.growthbook_refresh_interval = 60

            result = init_growthbook()
            assert result is not None
            assert result.is_feature_on("test-flag", {"userId": "1"}) is True

        shutdown_growthbook()
