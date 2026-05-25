"""
Full Analysis Orchestrator
============================
Page-aware routing system with harmony response format.

Harmony Response:
- All agents return: {sections, nsr_warnings, engine, raw}
- Each section has title + flowing content (ChatGPT style)
- Intelligent: only returns sections relevant to user's request
"""

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
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in QUOTATION_TRIGGERS)


# ── Intelligent Section Filter ───────────────────

SECTION_INTENT_MAP = {
    # Sales bot
    "client_summary":          ["client summary", "summary", "client overview", "who is the client"],
    "technical_reality_check": ["reality check", "technical check", "is it possible", "feasibility"],
    "red_flags":               ["red flag", "warning", "risk", "problem", "issue", "concern"],
    "suggested_reply":         ["reply", "response", "what to say", "message draft", "suggested reply"],

    # Service guide
    "service_description":     ["what is", "service description", "overview"],
    "how_it_works":            ["how it works", "how does it work", "process"],
    "real_challenges":         ["challenge", "difficulty", "problem"],
    "technical_limitations":   ["limitation", "limit", "constraint", "restriction"],
    "recommended_stack":       ["tech stack", "technology", "tools", "recommended"],

    # Quotation
    "project_overview":        ["project overview", "overview"],
    "phase_breakdown":         ["phase", "breakdown", "steps", "timeline"],
    "tech_stack":              ["tech stack", "technology"],
    "total_summary":           ["total", "price", "cost", "budget", "how much"],
    "client_requirements":     ["client provides", "requirement", "what client"],
    "additional_notes":        ["notes", "additional", "extra"],

    # Alternative guide
    "problem_identified":      ["problem", "issue", "what's wrong"],
    "real_world_explanation":  ["explanation", "explain", "why"],
    "best_alternative":        ["best alternative", "best option", "recommended"],
    "budget_alternative":      ["budget", "cheap", "affordable", "low cost"],
    "client_message_draft":    ["client message", "how to explain", "tell client"],
}


def _filter_sections(sections: list, user_message: str) -> list:
    """
    Return only sections relevant to user's request.
    If no specific intent detected, return all sections.
    """
    msg_lower = user_message.lower()

    matched_ids = set()
    for section_id, keywords in SECTION_INTENT_MAP.items():
        if any(kw in msg_lower for kw in keywords):
            matched_ids.add(section_id)

    if matched_ids:
        filtered = [s for s in sections if s["id"] in matched_ids]
        if filtered:
            return filtered

    return sections


# ── Harmony Response Builder ─────────────────────

def _build_harmony_response(sections: list) -> str:
    """
    Build a single flowing raw response from sections.
    ChatGPT style: **Title** then content, step by step.
    """
    parts = []
    for section in sections:
        title   = section.get("title", "")
        content = section.get("content", "").strip()
        if content:
            parts.append(f"**{title}**\n{content}")
    return "\n\n".join(parts)


# ── Main Entry ──────────────────────────────────

def run_full_analysis(
    conversation_text: str,
    conversation_history: list = None,
    is_follow_up: bool = False,
    model_key: str = None,
    page_context: str = None,
) -> dict:
    has_history    = bool(conversation_history) or is_follow_up
    client_context = _extract_client_context(conversation_history)

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

    # No page context: use intent classifier
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
    if not history:
        result         = sales_bot.run(conversation=text, model_key=model_key or "claude-sonnet")
        input_warnings = nsr_check(text)
        filtered       = _filter_sections(result["sections"], text)
        harmony_raw    = _build_harmony_response(filtered)

        return {
            "agent":                 "sales_bot",
            "intent":                "sales_analysis",
            "mode":                  "analysis",
            "page":                  "sales_bot",
            "engine":                result.get("engine"),
            "sections":              filtered,
            "nsr_warnings":          input_warnings,
            "combined_nsr_warnings": input_warnings,
            "raw":                   harmony_raw,
        }

    return _run_conversation(
        text, history, client_context,
        intent="sales_followup",
        model_key=model_key,
        page="sales_bot",
    )


def _handle_service_guide_page(
    text: str, history: list, client_context: str, model_key: str
) -> dict:
    if _is_quotation_request(text):
        service_desc   = client_context if client_context else text
        result         = service_guide.quotation(
            service_description=service_desc,
            model_key=model_key or "claude-sonnet",
        )
        input_warnings = nsr_check(service_desc + " " + text)
        filtered       = _filter_sections(result["sections"], text)
        harmony_raw    = _build_harmony_response(filtered)

        return {
            "agent":                 "quotation",
            "intent":                "quotation_generation",
            "mode":                  "quotation",
            "page":                  "service_guide",
            "engine":                result.get("engine"),
            "sections":              filtered,
            "nsr_warnings":          input_warnings,
            "combined_nsr_warnings": input_warnings,
            "raw":                   harmony_raw,
        }

    if not history:
        result         = service_guide.guide(
            service_description=text,
            model_key=model_key or "claude-sonnet",
        )
        input_warnings = nsr_check(text)
        filtered       = _filter_sections(result["sections"], text)
        harmony_raw    = _build_harmony_response(filtered)

        return {
            "agent":                 "service_guide",
            "intent":                "service_guide",
            "mode":                  "guide",
            "page":                  "service_guide",
            "engine":                result.get("engine"),
            "sections":              filtered,
            "nsr_warnings":          input_warnings,
            "combined_nsr_warnings": input_warnings,
            "raw":                   harmony_raw,
        }

    return _run_conversation(
        text, history, client_context,
        intent="tech_discussion",
        model_key=model_key,
        page="service_guide",
    )


def _handle_alternative_guide_page(
    text: str, history: list, client_context: str, model_key: str
) -> dict:
    if not history:
        result         = alternative_guide.run(
            problem_description=text,
            model_key=model_key or "claude-sonnet",
        )
        input_warnings = nsr_check(text)
        filtered       = _filter_sections(result["sections"], text)
        harmony_raw    = _build_harmony_response(filtered)

        return {
            "agent":                 "alternative_guide",
            "intent":                "alternative_guide",
            "mode":                  "alternative",
            "page":                  "alternative_guide",
            "engine":                result.get("engine"),
            "sections":              filtered,
            "nsr_warnings":          input_warnings,
            "combined_nsr_warnings": input_warnings,
            "raw":                   harmony_raw,
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
    has_history = bool(history)
    intent      = classify(text, has_history=has_history)

    return _run_conversation(
        text, history, client_context,
        intent=intent,
        model_key=model_key,
        page="system_prompt",
    )


# ── Helper Runners ──────────────────────────────

def _run_full_three_agents(text: str, model_key: str) -> dict:
    """Run all 3 agents in parallel, merge into harmony response."""
    input_warnings = nsr_check(text)
    chosen         = model_key or "claude-sonnet"

    with ThreadPoolExecutor(max_workers=3) as executor:
        sf = executor.submit(sales_bot.run, text, chosen)
        gf = executor.submit(service_guide.guide, text, chosen)
        af = executor.submit(alternative_guide.run, text, chosen)
        sr, gr, ar = sf.result(), gf.result(), af.result()

    all_sections = sr["sections"] + gr["sections"] + ar["sections"]
    filtered     = _filter_sections(all_sections, text)
    harmony_raw  = _build_harmony_response(filtered)

    return {
        "agent":                 "full_analysis",
        "intent":                "sales_analysis",
        "mode":                  "analysis",
        "engine":                sr.get("engine"),
        "sections":              filtered,
        "nsr_warnings":          input_warnings,
        "combined_nsr_warnings": input_warnings,
        "raw":                   harmony_raw,
        "agents": [
            {"agent_name": "sales_bot",         "agent_title": "Fiverr Sales Bot",
             "engine": sr.get("engine"), "sections": sr["sections"]},
            {"agent_name": "service_guide",     "agent_title": "Service Guide",
             "engine": gr.get("engine"), "sections": gr["sections"]},
            {"agent_name": "alternative_guide", "agent_title": "Alternative Guide",
             "engine": ar.get("engine"), "sections": ar["sections"]},
        ],
    }


def _run_conversation(
    text: str, history: list, client_context: str,
    intent: str, model_key: str, page: str = None,
) -> dict:
    result      = conversation.run(
        user_message=text,
        conversation_history=history or [],
        client_context=client_context,
        intent=intent,
        model_key=model_key,
    )
    filtered    = _filter_sections(result["sections"], text)
    harmony_raw = _build_harmony_response(filtered)

    return {
        "agent":                 "full_analysis",
        "intent":                intent,
        "mode":                  "conversation",
        "page":                  page,
        "engine":                result["engine"],
        "response":              result["response"],
        "sections":              filtered,
        "nsr_warnings":          result.get("nsr_warnings", []),
        "combined_nsr_warnings": result.get("nsr_warnings", []),
        "raw":                   harmony_raw,
    }


def _extract_client_context(history: list) -> str:
    if not history:
        return ""
    for msg in history:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""