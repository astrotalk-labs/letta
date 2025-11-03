from typing import Optional
from anthropic import AnthropicVertex

from letta.settings import model_settings


class AnthropicVertexClient:
    """Client for Claude models on Vertex AI using Anthropic's official SDK"""

    def __init__(self, project_id: Optional[str] = None, region: Optional[str] = None):
        self.project_id = project_id or model_settings.google_cloud_project
        self.region = region or model_settings.google_cloud_location

        if not self.project_id or not self.region:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set for Vertex AI"
            )

        self.client = AnthropicVertex(
            project_id=self.project_id,
            region=self.region,
        )

    def _get_client(self):
        """Return the Anthropic Vertex client"""
        return self.client