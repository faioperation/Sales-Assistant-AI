"""
Vision Agent
============
Image analysis with RAG context injection.
Analyzes images and enriches with knowledge base data
so all downstream agents can use the combined context.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import OPENAI_API_KEY, LLM_MODEL
from rag.retriever import retrieve, format_context

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.4,
)

SYSTEM_PROMPT = """
You are a senior sales and technical analyst. You are analyzing images sent by clients or salespeople.

Your job is to extract ALL useful information from the image and provide a structured analysis:

1. IMAGE DESCRIPTION
   - What is shown in the image (screenshot, document, design, conversation, diagram, etc.)
   - Any visible text (OCR) — extract ALL text you can see

2. SALES CONTEXT
   - What is the client asking for or showing?
   - What service/product/project does this relate to?
   - Any budget, timeline, or scope hints visible?

3. TECHNICAL DETAILS
   - Technologies, platforms, tools mentioned or visible
   - System architecture or workflow if shown
   - Any technical requirements or constraints

4. KEY INSIGHTS
   - What does this tell us about the client's needs?
   - What questions should we ask next?
   - Any red flags or opportunities?

Be precise. Extract everything. Do NOT make up content.
"""


def analyze(image_data: str, user_question: str = "") -> dict:
    """
    Analyze image and enrich with RAG context.

    Args:
        image_data:    base64 string, data:image URL, or http URL
        user_question: optional user context/question about the image

    Returns:
        {
            "sections":        list,
            "nsr_warnings":    list,
            "raw":             str,   ← full analysis text for downstream agents
            "vision_context":  str,   ← clean text summary for agent injection
        }
    """
    if not image_data:
        return {
            "sections":       [{"id": "vision_error", "title": "Error",
                                "content": "No image provided"}],
            "nsr_warnings":   [],
            "raw":            "",
            "vision_context": "",
        }

    # Build image URL
    if image_data.startswith("http") or image_data.startswith("data:image"):
        image_url = image_data
    else:
        image_url = f"data:image/png;base64,{image_data}"

    question = user_question or "Analyze this image thoroughly for sales and technical context."

    # ── Step 1: Analyze the image ────────────────
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=[
            {"type": "text",      "text": question},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]),
    ]

    try:
        response     = llm.invoke(messages)
        vision_output = response.content
    except Exception as e:
        return {
            "sections":       [{"id": "vision_error", "title": "Error",
                                "content": f"Vision analysis failed: {str(e)}"}],
            "nsr_warnings":   [],
            "raw":            "",
            "vision_context": "",
            "error":          str(e),
        }

    # ── Step 2: Use vision output to fetch RAG context ───
    try:
        rag_query  = (user_question + " " + vision_output)[:500]
        chunks     = retrieve(query=rag_query, k=5)
        rag_context = format_context(chunks)
    except Exception as e:
        print(f"[Vision] RAG fetch failed (continuing without): {e}")
        rag_context = ""

    # ── Step 3: Build enriched vision context for agents ─
    vision_context_parts = [f"=== IMAGE ANALYSIS ===\n{vision_output}"]
    if rag_context and rag_context != "No relevant knowledge found.":
        vision_context_parts.append(f"=== RELEVANT KNOWLEDGE BASE ===\n{rag_context}")

    vision_context = "\n\n".join(vision_context_parts)

    sections = [
        {"id": "image_analysis", "title": "Image Analysis", "content": vision_output},
    ]
    if rag_context:
        sections.append({
            "id":      "rag_context",
            "title":   "Related Knowledge",
            "content": rag_context,
        })

    return {
        "sections":       sections,
        "nsr_warnings":   [],
        "raw":            vision_output,
        "vision_context": vision_context,
    }