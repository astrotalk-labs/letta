"""Hardcoded Bedrock inference profile IDs.

Only the profile ID is hardcoded here — partition, region, and account are taken at
runtime from the BEDROCK_INFERENCE_PROFILE_ARN env var prefix (see `_build_bedrock_arn`
in anthropic_client.py), so these IDs are portable across environments without
touching this file.
"""

# Model-name based routing: matched by substring against the requested model name.
MODEL_INFERENCE_PROFILES: dict = {
    "sonnet-5": "yfsj0hvyx1ls",
}

# Cohort-based routing: gated users get a dedicated inference profile per cohort.
COHORT_INFERENCE_PROFILES: dict = {
    "AT_NATIVE": "v9a7gf6hhoqm",
    "AT_FOREIGN": "ngef7rrnxwrh",
    "INDIAN_AT": "rg76qwszdgvo",
    "PANDITJI": "18c7gtd1mst5",
    "LUMUS": "9gx7yplss61x",
}
