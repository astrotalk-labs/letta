"""Hardcoded Bedrock inference profile IDs.

Only the profile ID is hardcoded here — partition, region, and account are taken at
runtime from the BEDROCK_INFERENCE_PROFILE_ARN env var prefix (see `_build_bedrock_arn`
in anthropic_client.py), so these IDs are portable across environments without
touching this file.
"""

# Profile used as the fallback target when the primary Bedrock call times out.
FALLBACK_INFERENCE_PROFILE = "gp9qehu2kfz1"

# Model-name based routing: matched by substring against the requested model name.
MODEL_INFERENCE_PROFILES: dict = {
    "sonnet-5": "yfsj0hvyx1ls",
}

# Cohort-based routing: gated users get a dedicated inference profile per cohort.
COHORT_INFERENCE_PROFILES: dict = {
    "AT_NATIVE": "8fzsefawomq9",
    "AT_FOREIGN": "ig3s2vdheodo",
    "INDIAN_AT": "o1vgfp9mx4eu",
    "PANDITJI": "4at9g7c6ugh9",
    "LUMUS": "73kcqlfiwgc4",
    "LETTA_CONTROL_VARIATION": "n4y5wim2o5gx",
    "LETTA_EXPERIMENT_VARIATION": "4qkesixe4j2v",
}
