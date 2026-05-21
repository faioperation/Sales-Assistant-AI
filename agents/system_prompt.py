"""
System Prompt Agent - Free-form advisor
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rag.retriever import retrieve, format_context
from nsr.rules import check
from config import OPENAI_API_KEY, LLM_MODEL

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.5,
)

SYSTEM_PROMPT = """
You are a smart sales and technical advisor for a freelance agency.

Give direct, practical, actionable advice. Be honest about limitations.
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

    response   = llm.invoke(messages)
    output     = response.content
    violations = check(output)

    sections = [
        {"id": "ai_advisor_response", "title": "AI Advisor Response", "content": output},
    ]

    return {
        "sections":     sections,
        "nsr_warnings": violations,
        "raw":          output,
    }