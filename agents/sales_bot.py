"""
Fiverr Sales Bot Agent
Uses Claude primary, OpenAI fallback if quota exceeded.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from rag.retriever import retrieve, format_context
from nsr.rules import check
from agents.llm_provider import invoke_with_fallback, format_engine_notice


SYSTEM_PROMPT = """
You are an expert Fiverr sales assistant helping a seller analyze client conversations.

Read the conversation and return a structured analysis with 4 sections:

1. CLIENT SUMMARY
2. TECHNICAL REALITY CHECK
3. RED FLAGS
4. SUGGESTED REPLY

Use the KNOWLEDGE BASE for accurate technical information.
Be honest about limitations. Do NOT make up facts.
"""


def run(conversation: str, model_key: str = "claude-sonnet") -> dict:
    chunks  = retrieve(query=conversation[:500], k=5)
    context = format_context(chunks)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
KNOWLEDGE BASE:
{context}

CLIENT CONVERSATION:
{conversation}

Analyze and provide the 4-section response.
"""),
    ]

    output, engine = invoke_with_fallback(messages, model_key=model_key, temperature=0.4)
    notice = format_engine_notice(engine)
    violations = check(output)

    parsed   = _parse_sections(output)
    sections = [
        {"id": "client_summary",          "title": "Client Summary",            "content": notice + parsed["summary"]},
        {"id": "technical_reality_check", "title": "Technical Reality Check",   "content": parsed["reality_check"]},
        {"id": "red_flags",               "title": "Red Flags",                 "content": parsed["red_flags"]},
        {"id": "suggested_reply",         "title": "Suggested Reply",           "content": parsed["reply_draft"]},
    ]

    return {
        "sections":     sections,
        "nsr_warnings": violations,
        "engine":       engine,
        "raw":          notice + output,
    }


def _parse_sections(text: str) -> dict:
    result = {
        "summary":       "",
        "reality_check": "",
        "red_flags":     "",
        "reply_draft":   "",
    }
    markers = {
        "summary":       ["1. CLIENT SUMMARY", "CLIENT SUMMARY"],
        "reality_check": ["2. TECHNICAL REALITY CHECK", "TECHNICAL REALITY CHECK"],
        "red_flags":     ["3. RED FLAGS", "RED FLAGS"],
        "reply_draft":   ["4. SUGGESTED REPLY", "SUGGESTED REPLY"],
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