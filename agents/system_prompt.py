"""
System Prompt Agent - Free-form advisor
Claude Sonnet primary, GPT-4o fallback.
Harmony response structure.
"""

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rag.retriever import retrieve, format_context
from nsr.rules import check
from agents.llm_provider import invoke_with_fallback, format_engine_notice


SYSTEM_PROMPT = """
You are a smart sales and technical advisor for a freelance agency.

Give direct, practical, actionable advice. Be honest about limitations.

Format your response clearly:
- Start with a short title line
- Then give a flowing, conversational explanation
- No unnecessary bullet lists unless comparing options
- Be concise unless detail is requested
"""


def run(user_input: str,
        document_content: str = "",
        image_description: str = "",
        conversation_history: list = None) -> dict:

    chunks  = retrieve(query=user_input, k=3)
    context = format_context(chunks)

    user_content = f"QUESTION:\n{user_input}"

    if document_content:
        user_content += f"\n\nDOCUMENT:\n{document_content[:3000]}"

    if image_description:
        user_content += f"\n\nIMAGE CONTEXT:\n{image_description}"

    if context and context != "No relevant knowledge found.":
        user_content += f"\n\nKNOWLEDGE BASE:\n{context}"

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    if conversation_history:
        for msg in conversation_history[-6:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_content))

    output, engine = invoke_with_fallback(
        messages,
        model_key="claude-sonnet",
        temperature=0.5,
    )
    notice     = format_engine_notice(engine)
    violations = check(output)
    final      = notice + output

    sections = [
        {"id": "ai_advisor_response", "title": "AI Advisor Response", "content": final},
    ]

    return {
        "sections":     sections,
        "nsr_warnings": violations,
        "engine":       engine,
        "raw":          final,
    }