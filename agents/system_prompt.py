"""
System Prompt Agent - Free-form advisor
Claude Sonnet primary, GPT-4o fallback.
Web search support via DuckDuckGo → OpenAI pipeline.
"""

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rag.retriever import retrieve, format_context
from nsr.rules import check
from agents.llm_provider import invoke_with_fallback, format_engine_notice
from rag.web_retriever import should_web_search, retrieve_and_analyze


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

    # ── Web Search Check ─────────────────────────
    if should_web_search(user_input):
        return _run_with_web_search(
            user_input=user_input,
            document_content=document_content,
            conversation_history=conversation_history,
        )

    # ── Normal RAG Flow ──────────────────────────
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

    return {
        "sections": [
            {"id": "ai_advisor_response", "title": "AI Advisor Response", "content": final},
        ],
        "nsr_warnings":    violations,
        "engine":          engine,
        "web_search_used": False,
        "raw":             final,
    }


# ── Web Search Flow ──────────────────────────────

def _run_with_web_search(
    user_input: str,
    document_content: str,
    conversation_history: list,
) -> dict:
    """
    Web search pipeline for latest/unknown information.
    DuckDuckGo → StackOverflow/Medium/Reddit → OpenAI analysis
    """
    print(f"[SystemPrompt] Web search triggered.")

    # Build extra context
    extra_parts = []

    if document_content:
        extra_parts.append(f"DOCUMENT CONTEXT:\n{document_content[:1500]}")

    if conversation_history:
        recent = conversation_history[-4:]
        history_text = "\n".join(
            f"{m.get('role', '').upper()}: {m.get('content', '')}"
            for m in recent
            if m.get("content")
        )
        if history_text:
            extra_parts.append(f"RECENT CONVERSATION:\n{history_text}")

    extra_context = "\n\n".join(extra_parts)

    # Run web retrieval + analysis
    web_result = retrieve_and_analyze(
        user_query=user_input,
        extra_context=extra_context,
    )

    answer  = web_result.get("answer", "")
    sources = web_result.get("sources", [])

    # Format sources
    sources_text = ""
    if sources:
        source_lines = ["\n\n**Sources:**"]
        for s in sources:
            source_lines.append(f"- [{s['source_type']}] {s['title']} → {s['url']}")
        sources_text = "\n".join(source_lines)

    final = answer + sources_text
    violations = check(user_input + " " + answer)

    return {
        "sections": [
            {
                "id":      "web_search_result",
                "title":   "Web Search Result",
                "content": final,
            },
        ],
        "nsr_warnings":    violations,
        "engine":          "openai_web_search",
        "web_search_used": True,
        "sources":         sources,
        "raw":             final,
    }