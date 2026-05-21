"""
Full Analysis Orchestrator
============================
Page-aware routing system:

Frontend sends 'page_context' to indicate which page user is on:
- 'sales_bot'         -> Fiverr Sales Bot agent (client analysis)
- 'service_guide'     -> Service Guide agent OR Quotation generator
- 'alternative_guide' -> Alternative Guide agent
- 'system_prompt'     -> Free-form chat (or auto-routing)

Within a page, agent decides between specific actions:
- Service Guide page: detects 'quotation' keywords -> quotation mode
- All pages: handles random follow-up chat with context
"""

import re
from concurrent.futures import ThreadPoolExecutor
from agents import sales_bot, service_guide, alternative_guide, conversation
from nsr.rules import check as nsr_check
from orchestrator.intent_classifier import classify


# ── Quotation Detection ──────────────────────────

QUOTATION_TRIGGERS = [
    "make quotation", "make a quotation", "create quotation",
    "create a quotation", "draft quotation", "draft a quotation",
    "generate quotation", "build quotation", "build a quotation",
    "help me make the quotation", "help me with quotation",
    "quotation for the client", "quotation for client",
    "make proposal", "create proposal", "build proposal",
    "send quote", "give me a quote",
    "price breakdown", "cost breakdown",
]


def _is_quotation_request(text: str) -> bool:
    """Detect if user wants a quotation generated."""
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in QUOTATION_TRIGGERS)


# ── Main Entry ──────────────────────────────────

def run_full_analysis(
    conversation_text: str,
    conversation_history: list = None,
    is_follow_up: bool = False,
    model_key: str = None,
    page_context: str = None,
) -> dict:
    """
    Route based on frontend page + user intent.

    Args:
        conversation_text:    User's current message
        conversation_history: Previous chat turns
        is_follow_up:         Force follow-up mode
        model_key:            Backend-decided model
        page_context:         Current frontend page
    """
    has_history    = bool(conversation_history) or is_follow_up
    client_context = _extract_client_context(conversation_history)

    # ── Route by page_context ─────────────────────────

    if page_context == "sales_bot":
        return _handle_sales_bot_page(
            conversation_text, conversation_history, client_context, model_key
        )

    if page_context == "service_guide":
        return _handle_service_guide_page(
            conversation_text, conversation_history, client_context, model_key
        )

    if page_context == "alternative_guide":
        return _handle_alternative_guide_page(
            conversation_text, conversation_history, client_context, model_key
        )

    if page_context == "system_prompt":
        return _handle_system_prompt_page(
            conversation_text, conversation_history, client_context, model_key
        )

    # ── No page context: use intent classifier (legacy) ──
    intent = classify(conversation_text, has_history=has_history)

    if intent == "sales_analysis":
        return _run_full_three_agents(conversation_text, model_key)

    return _run_conversation(
        conversation_text, conversation_history, client_context, intent, model_key
    )


# ── Page Handlers ───────────────────────────────

def _handle_sales_bot_page(
    text: str, history: list, client_context: str, model_key: str
) -> dict:
    """
    Sales Bot page:
    - First message (no history): Run sales_bot agent for client analysis
    - Follow-up: Conversational (reply suggestions, strategy)
    """
    if not history:
        # First message - run sales bot analysis
        result = sales_bot.run(conversation=text, model_key=model_key or "claude-sonnet")
        input_warnings = nsr_check(text)

        return {
            "agent":  "sales_bot",
            "intent": "sales_analysis",
            "mode":   "analysis",
            "page":   "sales_bot",
            "engine": result.get("engine"),
            "sections":     result["sections"],
            "nsr_warnings": input_warnings,
            "combined_nsr_warnings": input_warnings,
        }

    # Follow-up - conversational mode
    return _run_conversation(
        text, history, client_context,
        intent="sales_followup",
        model_key=model_key,
        page="sales_bot",
    )


def _handle_service_guide_page(
    text: str, history: list, client_context: str, model_key: str
) -> dict:
    """
    Service Guide page:
    - Detect QUOTATION request -> run quotation generator
    - First service question (no history): Run service_guide agent
    - Follow-up: Conversational with tech context
    """
    # PRIORITY: Quotation detection
    if _is_quotation_request(text):
        # Use client_context as service description if available
        service_desc = client_context if client_context else text

        result = service_guide.quotation(
            service_description=service_desc,
            model_key=model_key or "claude-sonnet",
        )
        input_warnings = nsr_check(service_desc + " " + text)

        return {
            "agent":  "quotation",
            "intent": "quotation_generation",
            "mode":   "quotation",
            "page":   "service_guide",
            "engine": result.get("engine"),
            "sections":     result["sections"],
            "nsr_warnings": input_warnings,
            "combined_nsr_warnings": input_warnings,
        }

    # First service guide question - run service guide agent
    if not history:
        result = service_guide.guide(
            service_description=text,
            model_key=model_key or "claude-sonnet",
        )
        input_warnings = nsr_check(text)

        return {
            "agent":  "service_guide",
            "intent": "service_guide",
            "mode":   "guide",
            "page":   "service_guide",
            "engine": result.get("engine"),
            "sections":     result["sections"],
            "nsr_warnings": input_warnings,
            "combined_nsr_warnings": input_warnings,
        }

    # Follow-up - conversational with tech context
    return _run_conversation(
        text, history, client_context,
        intent="tech_discussion",
        model_key=model_key,
        page="service_guide",
    )


def _handle_alternative_guide_page(
    text: str, history: list, client_context: str, model_key: str
) -> dict:
    """
    Alternative Guide page:
    - First message: Run alternative_guide agent
    - Follow-up: Conversational
    """
    if not history:
        result = alternative_guide.run(
            problem_description=text,
            model_key=model_key or "claude-sonnet",
        )
        input_warnings = nsr_check(text)

        return {
            "agent":  "alternative_guide",
            "intent": "alternative_guide",
            "mode":   "alternative",
            "page":   "alternative_guide",
            "engine": result.get("engine"),
            "sections":     result["sections"],
            "nsr_warnings": input_warnings,
            "combined_nsr_warnings": input_warnings,
        }

    return _run_conversation(
        text, history, client_context,
        intent="sales_followup",
        model_key=model_key,
        page="alternative_guide",
    )


def _handle_system_prompt_page(
    text: str, history: list, client_context: str, model_key: str
) -> dict:
    """
    System Prompt page: Free-form chat with auto intent detection.
    """
    has_history = bool(history)
    intent = classify(text, has_history=has_history)

    return _run_conversation(
        text, history, client_context,
        intent=intent,
        model_key=model_key,
        page="system_prompt",
    )


# ── Helper Runners ──────────────────────────────

def _run_full_three_agents(text: str, model_key: str) -> dict:
    """Run all 3 sales agents in parallel."""
    input_warnings = nsr_check(text)
    chosen = model_key or "claude-sonnet"

    with ThreadPoolExecutor(max_workers=3) as executor:
        sf = executor.submit(sales_bot.run, text, chosen)
        gf = executor.submit(service_guide.guide, text, chosen)
        af = executor.submit(alternative_guide.run, text, chosen)
        sr, gr, ar = sf.result(), gf.result(), af.result()

    return {
        "agent":  "full_analysis",
        "intent": "sales_analysis",
        "mode":   "analysis",
        "model_requested": chosen,
        "agents": [
            {"agent_name": "sales_bot",         "agent_title": "Fiverr Sales Bot",
             "engine": sr.get("engine"), "sections": sr["sections"],
             "nsr_warnings": input_warnings},
            {"agent_name": "service_guide",     "agent_title": "Service Guide",
             "engine": gr.get("engine"), "sections": gr["sections"],
             "nsr_warnings": input_warnings},
            {"agent_name": "alternative_guide", "agent_title": "Alternative Guide",
             "engine": ar.get("engine"), "sections": ar["sections"],
             "nsr_warnings": input_warnings},
        ],
        "combined_nsr_warnings": input_warnings,
    }


def _run_conversation(
    text: str, history: list, client_context: str,
    intent: str, model_key: str, page: str = None,
) -> dict:
    """Run conversational agent with given intent."""
    result = conversation.run(
        user_message=text,
        conversation_history=history or [],
        client_context=client_context,
        intent=intent,
        model_key=model_key,
    )

    return {
        "agent":  "full_analysis",
        "intent": intent,
        "mode":   "conversation",
        "page":   page,
        "engine": result["engine"],
        "response":     result["response"],
        "sections":     result["sections"],
        "nsr_warnings": result.get("nsr_warnings", []),
        "combined_nsr_warnings": result.get("nsr_warnings", []),
    }


def _extract_client_context(history: list) -> str:
    if not history:
        return ""
    for msg in history:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""