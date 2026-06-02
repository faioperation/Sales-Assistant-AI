"""
Full Analysis Orchestrator
============================
Page-aware routing system with harmony response format.

Harmony Response:
- All agents return: {sections, nsr_warnings, engine, raw}
- Each section has title + flowing content (ChatGPT style)
- Intelligent: only returns sections relevant to user's request

Vision Flow:
- If image_data is provided, vision agent runs FIRST
- Vision output + RAG context is injected into all downstream agents
- All agents (sales_bot, service_guide, alternative_guide, conversation)
  receive the enriched vision context automatically
"""

from concurrent.futures import ThreadPoolExecutor
from agents import sales_bot, service_guide, alternative_guide, conversation
from agents import vision as vision_agent
from nsr.rules import check as nsr_check
from orchestrator.intent_classifier import classify
from rag.web_retriever import should_web_search, retrieve_and_analyze


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

    # Vision
    "image_analysis":          ["image", "screenshot", "photo", "picture", "what is this", "analyze image"],
}


def _filter_sections(sections: list, user_message: str) -> list:
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
    parts = []
    for section in sections:
        title   = section.get("title", "")
        content = section.get("content", "").strip()
        if content:
            parts.append(f"**{title}**\n{content}")
    return "\n\n".join(parts)


# ── Vision Pre-Processing ────────────────────────

def _run_vision_if_needed(
    image_data: str,
    user_input: str,
) -> tuple[str, list]:
    """
    Run vision agent if image_data is present.

    Returns:
        (vision_context, vision_sections)
        vision_context: enriched text to inject into all agents
        vision_sections: sections to prepend to final response
    """
    if not image_data:
        return "", []

    print("[FullAnalysis] Image detected — running vision agent first...")
    vision_result = vision_agent.analyze(
        image_data=image_data,
        user_question=user_input,
    )

    vision_context  = vision_result.get("vision_context", vision_result.get("raw", ""))
    vision_sections = vision_result.get("sections", [])

    return vision_context, vision_sections


# ── Inject Vision Context into User Input ────────

def _enrich_input_with_vision(
    user_input: str,
    vision_context: str,
) -> str:
    """
    Prepend vision context to user input so all agents
    receive the full picture (image analysis + RAG).
    """
    if not vision_context:
        return user_input

    return (
        f"[IMAGE CONTEXT FROM VISION ANALYSIS]\n"
        f"{vision_context}\n\n"
        f"[USER REQUEST]\n"
        f"{user_input}"
    )


# ── Main Entry ──────────────────────────────────

def run_full_analysis(
    conversation_text: str,
    conversation_history: list = None,
    is_follow_up: bool = False,
    model_key: str = None,
    page_context: str = None,
    image_data: str = "",
) -> dict:
    has_history    = bool(conversation_history) or is_follow_up
    client_context = _extract_client_context(conversation_history)

    # ── Step 1: Vision pre-processing ────────────
    vision_context, vision_sections = _run_vision_if_needed(image_data, conversation_text)

    # Enrich the conversation text with vision output
    # so ALL agents automatically get image context
    enriched_text = _enrich_input_with_vision(conversation_text, vision_context)

    # ── Step 2: Route to appropriate agent ───────
    if page_context == "sales_bot":
        result = _handle_sales_bot_page(
            enriched_text, conversation_history, client_context, model_key
        )

    elif page_context == "service_guide":
        result = _handle_service_guide_page(
            enriched_text, conversation_history, client_context, model_key
        )

    elif page_context == "alternative_guide":
        result = _handle_alternative_guide_page(
            enriched_text, conversation_history, client_context, model_key
        )

    elif page_context == "system_prompt":
        result = _handle_system_prompt_page(
            enriched_text, conversation_history, client_context, model_key
        )

    else:
        intent = classify(enriched_text, has_history=has_history)

        # Web search check — even for sales_analysis intent
        if should_web_search(enriched_text):
            result = _run_web_search_flow(
                enriched_text, conversation_history, client_context, intent
            )
        elif intent == "sales_analysis":
            result = _run_full_three_agents(enriched_text, model_key)
        else:
            result = _run_conversation(
                enriched_text, conversation_history, client_context, intent, model_key
            )

    # ── Step 3: Prepend vision sections to response ─
    if vision_sections:
        result["sections"]      = vision_sections + result.get("sections", [])
        result["vision_used"]   = True
        result["raw"]           = _build_harmony_response(result["sections"])
    else:
        result["vision_used"] = False

    return result


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


def _run_web_search_flow(
    text: str, history: list, client_context: str, intent: str
) -> dict:
    """
    Web search pipeline for any intent when user requests latest/search info.
    Works for sales_analysis, sales_followup, tech_discussion, general_chat.
    """
    print(f"[FullAnalysis] Web search triggered for intent: {intent}")

    extra_parts = []
    if client_context:
        extra_parts.append(f"CLIENT CONTEXT:\n{client_context}")
    if history:
        recent = history[-4:]
        history_text = "\n".join(
            f"{m.get('role','').upper()}: {m.get('content','')}"
            for m in recent if m.get("content")
        )
        if history_text:
            extra_parts.append(f"RECENT CONVERSATION:\n{history_text}")

    extra_context = "\n\n".join(extra_parts)

    web_result = retrieve_and_analyze(
        user_query=text,
        extra_context=extra_context,
    )

    answer  = web_result.get("answer", "")
    sources = web_result.get("sources", [])

    sources_text = ""
    if sources:
        source_lines = ["\n\n**Sources:**"]
        for s in sources:
            source_lines.append(f"- [{s['source_type']}] {s['title']} -> {s['url']}")
        sources_text = "\n".join(source_lines)

    final      = answer + sources_text
    violations = nsr_check(text + " " + answer)

    return {
        "agent":                 "web_search",
        "intent":                intent,
        "mode":                  "web_search",
        "engine":                "openai_web_search",
        "web_search_used":       True,
        "sources":               sources,
        "sections": [
            {
                "id":      "web_search_result",
                "title":   "Web Search Result",
                "content": final,
            }
        ],
        "nsr_warnings":          violations,
        "combined_nsr_warnings": violations,
        "raw":                   final,
    }

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