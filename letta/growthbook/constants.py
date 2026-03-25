"""GrowthBook constants: standard attribute keys.

Add feature keys here as you define them in the GrowthBook dashboard.
"""


class GrowthBookAttributes:
    """Standard attribute keys used when building GrowthBook context."""

    USER_ID = "userId"
    AGENT_ID = "agentId"
    ORGANIZATION_ID = "organizationId"
    ENVIRONMENT = "environment"
    MODEL_ENDPOINT_TYPE = "modelEndpointType"
    MODEL = "model"


class GrowthBookFeatureKeys:
    """Registry of GrowthBook feature flag keys.

    Add new keys here as you create them in the GrowthBook dashboard,
    so every caller references a single source of truth.

    Example:
        MY_FEATURE = "my-feature-key"
    """

    USE_AZURE_EMBEDDINGS = "use_azure_embeddings"
