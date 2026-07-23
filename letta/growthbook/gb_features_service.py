"""Service responsible for fetching and caching GrowthBook feature definitions.

Mirrors the Java GBFeaturesService: periodically refreshes feature JSON from
the GrowthBook API (or loads it from a local JSON file as fallback).
"""

import json
import threading
import time
from pathlib import Path
from typing import Optional

import httpx

from letta.log import get_logger

logger = get_logger(__name__)

# How often to refresh features from the API (seconds)
_DEFAULT_REFRESH_INTERVAL = 60


class GBFeaturesService:
    """Manages GrowthBook feature definitions (the JSON blob).

    Supports two modes:
    1. **API mode** – polls ``{api_host}/api/features/{client_key}`` on a background thread.
    2. **Local mode** – reads feature definitions from a JSON file on disk.

    If both ``client_key`` and ``features_json_path`` are provided, the API is
    primary and the local file is used as a cold-start / fallback cache.
    """

    def __init__(
        self,
        client_key: Optional[str] = None,
        api_host: str = "https://cdn.growthbook.io",
        features_json_path: Optional[str] = None,
        refresh_interval: int = _DEFAULT_REFRESH_INTERVAL,
    ):
        self._client_key = client_key
        self._api_host = api_host.rstrip("/")
        self._features_json_path = features_json_path
        self._refresh_interval = refresh_interval

        self._features_json: str = "{}"
        self._lock = threading.Lock()
        self._refresh_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Load initial features
        self._load_initial_features()

        # Start background refresh if API mode
        if self._client_key:
            self._start_refresh_thread()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_features_json(self) -> str:
        """Return the current feature definitions JSON string."""
        with self._lock:
            return self._features_json

    def shutdown(self):
        """Stop the background refresh thread (call on app shutdown)."""
        self._stop_event.set()
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=5)
            logger.info("GrowthBook features refresh thread stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_initial_features(self):
        """Load features from local file first, then try API."""
        # Try local file first (cold-start cache)
        if self._features_json_path:
            self._load_from_file()

        # Try API if available
        if self._client_key:
            try:
                self._fetch_from_api()
            except Exception as e:
                logger.warning(f"Failed to fetch initial GrowthBook features from API: {e}")
                if self._features_json == "{}":
                    logger.warning("No cached features available – starting with empty feature set.")

    def _load_from_file(self):
        """Load features from a local JSON file."""
        try:
            path = Path(self._features_json_path)
            if path.exists():
                raw = path.read_text()
                # Validate it's proper JSON
                parsed = json.loads(raw)
                # The file should contain the features object directly
                # (same format as the "features" key in the API response)
                with self._lock:
                    self._features_json = json.dumps(parsed) if isinstance(parsed, dict) else raw
                logger.info(f"Loaded GrowthBook features from {self._features_json_path}")
            else:
                logger.warning(f"GrowthBook features file not found: {self._features_json_path}")
        except Exception as e:
            logger.error(f"Error loading GrowthBook features from file: {e}")

    def _fetch_from_api(self):
        """Fetch feature definitions from the GrowthBook API."""
        url = f"{self._api_host}/api/features/{self._client_key}"
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        features = data.get("features", {})
        features_json = json.dumps(features)

        with self._lock:
            self._features_json = features_json

        # Optionally persist to file as cache
        if self._features_json_path:
            try:
                Path(self._features_json_path).write_text(features_json)
            except Exception as e:
                logger.warning(f"Failed to cache GrowthBook features to file: {e}")

        logger.debug("Refreshed GrowthBook features from API.")

    def _start_refresh_thread(self):
        """Start a daemon thread that periodically refreshes features."""
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            name="growthbook-features-refresh",
            daemon=True,
        )
        self._refresh_thread.start()
        logger.info(f"GrowthBook features refresh thread started (interval={self._refresh_interval}s).")

    def _refresh_loop(self):
        """Background loop that fetches features periodically."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._refresh_interval)
            if self._stop_event.is_set():
                break
            try:
                self._fetch_from_api()
            except Exception as e:
                logger.warning(f"GrowthBook features refresh failed: {e}")
