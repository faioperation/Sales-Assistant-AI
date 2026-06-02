"""
Conversational Agent (Dual Engine with Auto-Fallback)
========================================================
Smart conversation with engine selection based on intent.

Web Search Feature:
- Triggered when user uses search keywords
- Searches StackOverflow, Medium, Reddit via DuckDuckGo
- Analyzed by OpenAI GPT-4o
- Works for ALL intents including sales_analysis
"""

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rag.retriever import retrieve, format_context
from nsr.rules import check
from agents.llm_provider import invoke_with_fallback, format_engine_notice
from rag.web_retriever import should_web_search, retrieve_and_analyze


# ── System Prompts ───────────────────────────────

SALES_FOLLOWUP_PROMPT = """
You are a smart sales assistant for a freelance agency. You help the salesperson
handle Fiverr client conversations from start to finish.

YOUR ROLE:
- Draft professional replies to clients
- Create quotations and proposals
- Suggest follow-up questions to ask the client
- Provide strategic negotiation advice
- Reality-check client expectations

CONTEXT AWARENESS:
- Always reference the ORIGINAL CLIENT CONVERSATION when relevant
- Remember what was discussed in previous turns
- Connect current question to the broader sales situation

GUIDELINES:
- Be conversational, not formal
- Use KNOWLEDGE BASE for accurate technical/pricing info
- Do NOT hallucinate — say "I don't know" when uncertain
- Be concise unless detail is requested
- When drafting client messages, mark them clearly with quotes or labels
"""

TECH_DISCUSSION_PROMPT = """
You are a senior technical consultant helping a freelancer make business and
technical decisions.

YOUR ROLE:
- Explain tools, platforms, frameworks accurately
- Provide honest assessment of capabilities and limitations
- Share pricing benchmarks and timeline estimates
- Recommend tech stacks for specific use cases

GUIDELINES:
- Be precise and factual
- Use KNOWLEDGE BASE for verified information
- Always mention real-world limitations (rate limits, costs, scale issues, edge cases)
- Do NOT hallucinate features or capabilities
- Give concrete, actionable advice with examples
- Compare alternatives when relevant
"""

GENERAL_CHAT_PROMPT = """
You are a helpful AI assistant. Answer the user's question clearly and accurately.

GUIDELINES:
- For factual questions, provide accurate information
- Be concise by default — expand only when asked
- Use examples to clarify complex topics
- If unsure, acknowledge the uncertainty
- Stay friendly and conversational
"""

WEB_SEARCH_PROMPT = """
You are a smart sales and technical assistant with access to real-time web search results
from StackOverflow, Medium, and Reddit.

Your job:
- Use the web search results to answer the user's question accurately
- Connect findings to their sales or technical situation
- Mention sources where relevant (StackOverflow answer, Medium article, Reddit discussion)
- Be honest if results are not fully relevant

Always provide practical, actionable insights based on what was found.
"""


# ── Intent Config ────────────────────────────────

INTENT_CONFIG = {
    "sales_followup": {
        "default_model":      "claude-sonnet",
        "prompt":             SALES_FOLLOWUP_PROMPT,
        "use_rag":            True,
        "use_client_context": True,
        "temperature":        0.5,
    },
    "tech_discussion": {
        "default_model":      "claude-sonnet",
        "prompt":             TECH_DISCUSSION_PROMPT,
        "use_rag":            True,
        "use_client_context": False,
        "temperature":        0.4,
    },
    "general_chat": {
        "default_model":      "gpt-4o",
        "prompt":             GENERAL_CHAT_PROMPT,
        "use_rag":            False,
        "use_client_context": False,
        "temperature":        0.6,
    },
    # sales_analysis can also reach here when web search is triggered
    "sales_analysis": {
        "default_model":      "claude-sonnet",
        "prompt":             SALES_FOLLOWUP_PROMPT,
        "use_rag":            True,
        "use_client_context": True,
        "temperature":        0.5,
    },
}


def _get_config(intent: str) -> dict:
    return INTENT_CONFIG.get(intent, INTENT_CONFIG["general_chat"])


# ── Main Function ────────────────────────────────

def run(user_message: str,
        conversation_history: list = None,
        client_context: str = "",
        intent: str = "general_chat",
        model_key: str = None) -> dict:

    cfg = _get_config(intent)

    # ── Web Search Check ─────────────────────────
    # If user wants latest/search info, bypass normal RAG
    # and use DuckDuckGo → fetch → OpenAI pipeline
    if should_web_search(user_message):
        return _run_with_web_search(
            user_message=user_message,
            conversation_history=conversation_history,
            client_context=client_context,
            intent=intent,
        )

    # ── Normal RAG Flow ──────────────────────────
    messages = [SystemMessage(content=cfg["prompt"])]

    # Add RAG knowledge base context
    if cfg["use_rag"]:
        try:
            chunks  = retrieve(query=user_message, k=3)
            context = format_context(chunks)
            if context and context != "No relevant knowledge found.":
                messages.append(SystemMessage(
                    content=f"RELEVANT KNOWLEDGE BASE:\n{context}"
                ))
        except Exception as e:
            print(f"[Conversation] RAG error (continuing without): {e}")

    # Add client context for sales follow-ups
    if cfg["use_client_context"] and client_context:
        messages.append(SystemMessage(
            content=f"ORIGINAL CLIENT CONVERSATION:\n{client_context}"
        ))

    # Add conversation history (last 10 turns)
    if conversation_history:
        for msg in conversation_history[-10:]:
            role    = msg.get("role")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_message))

    chosen_model = model_key or cfg["default_model"]

    output, engine = invoke_with_fallback(
        messages,
        model_key=chosen_model,
        temperature=cfg["temperature"],
    )

    notice       = format_engine_notice(engine)
    final_output = notice + output

    combined_text = user_message + " " + output + " " + (client_context or "")
    violations    = check(combined_text)

    return {
        "response":        final_output,
        "engine":          engine,
        "intent":          intent,
        "web_search_used": False,
        "sections": [
            {"id": "ai_response", "title": "AI Response", "content": final_output},
        ],
        "nsr_warnings": violations,
    }


# ── Web Search Flow ──────────────────────────────

def _run_with_web_search(
    user_message: str,
    conversation_history: list,
    client_context: str,
    intent: str,
) -> dict:
    """
    Handle web search requests:
    DuckDuckGo search → fetch content → OpenAI analysis
    """
    print(f"[Conversation] Web search triggered for intent: {intent}")

    # Build extra context from conversation history + client context
    extra_parts = []

    if client_context:
        extra_parts.append(f"CLIENT CONTEXT:\n{client_context}")

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
        user_query=user_message,
        extra_context=extra_context,
    )

    answer  = web_result.get("answer", "")
    sources = web_result.get("sources", [])

    # Format sources as readable text
    sources_text = ""
    if sources:
        source_lines = ["\n\n**Sources:**"]
        for s in sources:
            source_lines.append(f"- [{s['source_type']}] {s['title']} → {s['url']}")
        sources_text = "\n".join(source_lines)

    final_output = answer + sources_text

    violations = check(user_message + " " + answer)

    return {
        "response":        final_output,
        "engine":          "openai_web_search",
        "intent":          intent,
        "web_search_used": True,
        "sources":         sources,
        "sections": [
            {
                "id":      "web_search_result",
                "title":   "Web Search Result",
                "content": final_output,
            },
        ],
        "nsr_warnings": violations,
    }