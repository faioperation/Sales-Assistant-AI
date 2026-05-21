"""
LLM Provider with Model Selection + Auto-Fallback
====================================================
Frontend chooses model, backend passes it here.

Flow:
1. User picks model (e.g., 'claude-haiku')
2. Provider tries chosen model
3. If quota fails -> automatically switches to equivalent model from other provider
   (claude-haiku  -> gpt-4o-mini    - both 'fast' tier)
   (claude-sonnet -> gpt-4o         - both 'powerful' tier)
"""

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY
from agents.model_registry import (
    MODELS, DEFAULT_MODEL,
    get_model_info, get_fallback_model,
)


# ── LLM Instance Cache ───────────────────────────
# Avoid recreating LLM clients for each request
_llm_cache = {}


def _get_llm(model_key: str, temperature: float):
    """Get or create an LLM instance for the given model key."""
    cache_key = f"{model_key}:{temperature}"
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    info = get_model_info(model_key)

    if info["provider"] == "anthropic":
        llm = ChatAnthropic(
            model=info["model_id"],
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=4096,
        )
    else:  # openai
        llm = ChatOpenAI(
            model=info["model_id"],
            api_key=OPENAI_API_KEY,
            temperature=temperature,
        )

    _llm_cache[cache_key] = llm
    return llm


# ── Quota/Rate-limit error detection ─────────────

QUOTA_ERROR_KEYWORDS = [
    "rate_limit", "rate limit", "quota", "exceeded",
    "insufficient_quota", "billing", "credit",
    "credit balance is too low",
    "overloaded", "capacity", "tokens per",
    "429",  # HTTP 429 too many requests
    "529",  # Anthropic overloaded
    "invalid_api_key", "authentication",
]


def _is_quota_error(error: Exception) -> bool:
    """Check if error is quota/rate-limit related (worth falling back)."""
    err_str = str(error).lower()
    return any(kw in err_str for kw in QUOTA_ERROR_KEYWORDS)


# ── Main fallback function ───────────────────────

def invoke_with_fallback(
    messages: list,
    model_key: str = None,
    primary: str = None,
    temperature: float = 0.4,
) -> tuple[str, str]:
    """
    Invoke LLM with automatic fallback on quota/rate-limit errors.

    Args:
        messages:    LangChain message list
        model_key:   Specific model from registry (e.g., 'claude-haiku')
                     If None, uses primary param or default model
        primary:     Legacy param - 'claude' or 'openai' (maps to default models)
        temperature: Sampling temperature

    Returns:
        (response_text, engine_used)
    """

    # Resolve which model to use
    if model_key is None:
        # Legacy support: primary='claude' -> claude-sonnet
        if primary == "claude":
            model_key = "claude-sonnet"
        elif primary == "openai":
            model_key = "gpt-4o"
        else:
            model_key = DEFAULT_MODEL

    # Try primary
    primary_info = get_model_info(model_key)
    primary_llm  = _get_llm(model_key, temperature)

    try:
        response = primary_llm.invoke(messages)
        return response.content, model_key

    except Exception as primary_err:
        if _is_quota_error(primary_err):
            print(f"[LLM] {model_key} quota exceeded, falling back...")
        else:
            print(f"[LLM] {model_key} error: {primary_err}")

        # Get fallback model from other provider, same tier
        fallback_key  = get_fallback_model(model_key)
        fallback_llm  = _get_llm(fallback_key, temperature)
        fallback_info = get_model_info(fallback_key)

        try:
            response = fallback_llm.invoke(messages)
            engine_used = f"{fallback_key}_fallback"
            print(f"[LLM] Fallback success: {fallback_key}")
            return response.content, engine_used

        except Exception as fallback_err:
            print(f"[LLM] Fallback {fallback_key} also failed: {fallback_err}")
            error_msg = (
                f"Both LLM engines failed.\n"
                f"Primary ({model_key}): {primary_err}\n"
                f"Fallback ({fallback_key}): {fallback_err}"
            )
            return error_msg, "error"


def format_engine_notice(engine: str) -> str:
    """
    Generate a notice when fallback was used.
    Empty string if primary worked.
    """
    if not engine.endswith("_fallback") and engine != "error":
        return ""

    if engine == "error":
        return "*Both LLM engines are unavailable*\n\n"

    # fallback case
    fallback_model = engine.replace("_fallback", "")
    info = get_model_info(fallback_model)
    provider_name = "OpenAI" if info["provider"] == "openai" else "Claude"

    return f"*Response by {provider_name} {info['display']} (fallback — primary model quota exceeded)*\n\n"