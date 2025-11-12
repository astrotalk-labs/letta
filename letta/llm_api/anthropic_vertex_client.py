from typing import Optional
from anthropic import AnthropicVertex
from letta.settings import model_settings
import os


class AnthropicVertexClient:
    """Client for Claude models on Vertex AI using Anthropic's official SDK"""

    def __init__(self, project_id: Optional[str] = None, region: Optional[str] = None):
        self.project_id = project_id or model_settings.google_cloud_project

        # Use provided region, env var, or default to global
        # For newer Claude models (claude-sonnet-4-5@20250929), use 'global'
        self.region = (
                region or
                os.getenv('GOOGLE_CLOUD_LOCATION') or
                os.getenv('ANTHROPIC_VERTEX_REGION') or
                'global'  # Default region for newer models
        )

        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT must be set for Vertex AI")

        self.client = AnthropicVertex(
            project_id=self.project_id,
            region=self.region,
        )

    def _get_client(self):
        """Return the Anthropic Vertex client"""
        return self.client