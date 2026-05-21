"""
Model Registry
================
Defines supported models for both Anthropic and OpenAI.
Frontend can choose any model, backend passes the choice to the AI layer.

Each model has:
- provider:   'anthropic' or 'openai'
- model_id:   actual API model identifier
- tier:       'fast' or 'powerful' (used for fallback matching)
- display:    human-readable name
"""

MODELS = {
    # ── Fast / Cheap tier ────────────────────────────
    "claude-haiku": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5",
        "tier":     "fast",
        "display":  "Claude Haiku (fast)",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "tier":     "fast",
        "display":  "GPT-4o Mini (fast)",
    },

    # ── Powerful tier ────────────────────────────────
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5",
        "tier":     "powerful",
        "display":  "Claude Sonnet (powerful)",
    },
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "tier":     "powerful",
        "display":  "GPT-4o (powerful)",
    },
}


# Default model when user doesn't specify
DEFAULT_MODEL = "claude-sonnet"


def get_model_info(model_key: str) -> dict:
    """Get model metadata. Falls back to default if not found."""
    return MODELS.get(model_key, MODELS[DEFAULT_MODEL])


def get_fallback_model(current_model_key: str) -> str:
    """
    Find a fallback model from the OTHER provider with the same tier.
    Example:
      claude-haiku  -> gpt-4o-mini  (both fast)
      gpt-4o        -> claude-sonnet (both powerful)
    """
    current = MODELS.get(current_model_key, MODELS[DEFAULT_MODEL])
    target_provider = "openai" if current["provider"] == "anthropic" else "anthropic"
    target_tier     = current["tier"]

    for key, info in MODELS.items():
        if info["provider"] == target_provider and info["tier"] == target_tier:
            return key

    # If no tier match, return any model from other provider
    for key, info in MODELS.items():
        if info["provider"] == target_provider:
            return key

    return DEFAULT_MODEL


def list_models() -> list:
    """List all available models for frontend dropdown."""
    return [
        {
            "key":      key,
            "display":  info["display"],
            "provider": info["provider"],
            "tier":     info["tier"],
        }
        for key, info in MODELS.items()
    ]