"""
Alternative Guide Agent
Uses Claude primary, OpenAI fallback if quota exceeded.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from rag.retriever import retrieve, format_context
from nsr.rules import check
from agents.llm_provider import invoke_with_fallback, format_engine_notice


SYSTEM_PROMPT = """
You are a senior technical architect helping find alternative solutions.

Provide a structured response with 5 sections:

1. PROBLEM IDENTIFIED
2. REAL-WORLD EXPLANATION
3. ALTERNATIVE SOLUTION 1 (Best Option)
4. ALTERNATIVE SOLUTION 2 (Budget-Friendly Option)
5. HOW TO EXPLAIN TO CLIENT

Always give REAL alternatives. Do NOT hallucinate tools.
"""


def run(problem_description: str, model_key: str = "claude-sonnet") -> dict:
    chunks  = retrieve(query=problem_description, k=5)
    context = format_context(chunks)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
KNOWLEDGE BASE:
{context}

PROBLEM:
{problem_description}

Provide alternative solutions.
"""),
    ]

    output, engine = invoke_with_fallback(messages, model_key=model_key, temperature=0.4)
    notice = format_engine_notice(engine)
    violations = check(output)

    parsed   = _parse_sections(output)
    sections = [
        {"id": "problem_identified",     "title": "Problem Identified",          "content": notice + parsed["problem"]},
        {"id": "real_world_explanation", "title": "Real-World Explanation",      "content": parsed["explanation"]},
        {"id": "best_alternative",       "title": "Best Alternative",            "content": parsed["alternative1"]},
        {"id": "budget_alternative",     "title": "Budget-Friendly Alternative", "content": parsed["alternative2"]},
        {"id": "client_message_draft",   "title": "Client Message Draft",        "content": parsed["client_message"]},
    ]

    return {
        "sections":     sections,
        "nsr_warnings": violations,
        "engine":       engine,
        "raw":          notice + output,
    }


def _parse_sections(text: str) -> dict:
    result = {
        "problem":        "",
        "explanation":    "",
        "alternative1":   "",
        "alternative2":   "",
        "client_message": "",
    }
    markers = {
        "problem":        ["1. PROBLEM IDENTIFIED", "PROBLEM IDENTIFIED"],
        "explanation":    ["2. REAL-WORLD EXPLANATION", "REAL-WORLD EXPLANATION"],
        "alternative1":   ["3. ALTERNATIVE SOLUTION 1", "ALTERNATIVE SOLUTION 1"],
        "alternative2":   ["4. ALTERNATIVE SOLUTION 2", "ALTERNATIVE SOLUTION 2"],
        "client_message": ["5. HOW TO EXPLAIN TO CLIENT", "HOW TO EXPLAIN TO CLIENT"],
    }
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