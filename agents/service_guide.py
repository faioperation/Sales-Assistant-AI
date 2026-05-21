"""
Service Guide Agent - Technical guide + quotation generator
Uses Claude primary, OpenAI fallback if quota exceeded.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from rag.retriever import retrieve, format_context
from nsr.rules import check
from agents.llm_provider import invoke_with_fallback, format_engine_notice


GUIDE_PROMPT = """
You are a senior technical consultant.

Provide a TECHNICAL SERVICE GUIDE with 5 sections:

1. WHAT IS THIS SERVICE
2. HOW IT WORKS
3. REAL-WORLD CHALLENGES
4. TECHNICAL LIMITATIONS
5. RECOMMENDED TECH STACK

Use KNOWLEDGE BASE for accuracy. Do NOT hallucinate.
"""

QUOTATION_PROMPT = """
You are an expert project manager creating a PROFESSIONAL QUOTATION.

Structure your response with these sections:

1. PROJECT OVERVIEW
2. PHASE BREAKDOWN
3. TECH STACK
4. TOTAL (days and price)
5. WHAT CLIENT PROVIDES
6. NOTES

Be REALISTIC. Use market rates. Do NOT underprice.
"""


def guide(service_description: str, model_key: str = "claude-sonnet") -> dict:
    chunks  = retrieve(query=service_description, k=5, section="technical_guide")
    context = format_context(chunks)

    messages = [
        SystemMessage(content=GUIDE_PROMPT),
        HumanMessage(content=f"""
KNOWLEDGE BASE:
{context}

SERVICE REQUEST:
{service_description}
"""),
    ]

    output, engine = invoke_with_fallback(messages, model_key=model_key, temperature=0.3)
    notice = format_engine_notice(engine)
    violations = check(output)

    parsed   = _parse_guide(output)
    sections = [
        {"id": "service_description",   "title": "Service Description",     "content": notice + parsed["what_is"]},
        {"id": "how_it_works",          "title": "How It Works",            "content": parsed["how_it_works"]},
        {"id": "real_challenges",       "title": "Real-World Challenges",   "content": parsed["challenges"]},
        {"id": "technical_limitations", "title": "Technical Limitations",   "content": parsed["limitations"]},
        {"id": "recommended_stack",     "title": "Recommended Tech Stack",  "content": parsed["tech_stack"]},
    ]

    return {
        "sections":     sections,
        "nsr_warnings": violations,
        "engine":       engine,
        "raw":          notice + output,
    }


def quotation(service_description: str,
              budget_hint: str = "",
              timeline_hint: str = "",
              model_key: str = "claude-sonnet") -> dict:
    chunks  = retrieve(query=service_description, k=5, section="pricing_timeline")
    context = format_context(chunks)

    extra = ""
    if budget_hint:
        extra += f"\nClient budget hint: {budget_hint}"
    if timeline_hint:
        extra += f"\nClient timeline hint: {timeline_hint}"

    messages = [
        SystemMessage(content=QUOTATION_PROMPT),
        HumanMessage(content=f"""
KNOWLEDGE BASE:
{context}

PROJECT:
{service_description}
{extra}
"""),
    ]

    output, engine = invoke_with_fallback(messages, model_key=model_key, temperature=0.3)
    notice = format_engine_notice(engine)
    violations = check(output)

    parsed   = _parse_quotation(output)
    sections = [
        {"id": "project_overview",     "title": "Project Overview",        "content": notice + parsed["overview"]},
        {"id": "phase_breakdown",      "title": "Phase Breakdown",         "content": parsed["phases"]},
        {"id": "tech_stack",           "title": "Tech Stack",              "content": parsed["tech_stack"]},
        {"id": "total_summary",        "title": "Total Days & Price",      "content": parsed["total"]},
        {"id": "client_requirements",  "title": "What Client Provides",    "content": parsed["client_provides"]},
        {"id": "additional_notes",     "title": "Additional Notes",        "content": parsed["notes"]},
    ]

    return {
        "sections":     sections,
        "nsr_warnings": violations,
        "engine":       engine,
        "raw":          notice + output,
    }


def _parse_guide(text: str) -> dict:
    result = {"what_is": "", "how_it_works": "", "challenges": "",
              "limitations": "", "tech_stack": ""}
    markers = {
        "what_is":       ["1. WHAT IS THIS SERVICE", "WHAT IS THIS SERVICE"],
        "how_it_works":  ["2. HOW IT WORKS", "HOW IT WORKS"],
        "challenges":    ["3. REAL-WORLD CHALLENGES", "REAL-WORLD CHALLENGES"],
        "limitations":   ["4. TECHNICAL LIMITATIONS", "TECHNICAL LIMITATIONS"],
        "tech_stack":    ["5. RECOMMENDED TECH STACK", "RECOMMENDED TECH STACK"],
    }
    return _extract(text, result, markers)


def _parse_quotation(text: str) -> dict:
    result = {"overview": "", "phases": "", "tech_stack": "",
              "total": "", "client_provides": "", "notes": ""}
    markers = {
        "overview":        ["1. PROJECT OVERVIEW", "PROJECT OVERVIEW"],
        "phases":          ["2. PHASE BREAKDOWN", "PHASE BREAKDOWN"],
        "tech_stack":      ["3. TECH STACK", "TECH STACK"],
        "total":           ["4. TOTAL", "TOTAL"],
        "client_provides": ["5. WHAT CLIENT PROVIDES", "WHAT CLIENT PROVIDES"],
        "notes":           ["6. NOTES", "NOTES"],
    }
    return _extract(text, result, markers)


def _extract(text: str, result: dict, markers: dict) -> dict:
    text_upper = text.upper()
    for key, possible_markers in markers.items():
        for marker in possible_markers:
            idx = text_upper.find(marker)
            if idx != -1:
                start = idx + len(marker)
                end   = len(text)
                for other_key, other_markers in markers.items():
                    if other_key == key:
                        continue
                    for om in other_markers:
                        oidx = text_upper.find(om, start)
                        if oidx != -1 and oidx < end:
                            end = oidx
                result[key] = text[start:end].strip()
                break
    return result