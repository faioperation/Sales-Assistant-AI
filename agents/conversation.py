"""
Conversational Agent (Dual Engine with Auto-Fallback)
========================================================
Smart conversation with engine selection based on intent:

- sales_followup  -> Claude primary (with client context + RAG)
- tech_discussion -> Claude primary (with RAG knowledge base)
- general_chat    -> OpenAI primary (no RAG, fast general knowledge)
- sales_analysis  -> Should not reach here (handled by full_analysis)

Features:
- Automatic engine fallback via centralized llm_provider
- Engine notice prepended when fallback is used
- Knowledge base retrieval only when relevant
- Conversation history (last 10 turns)
- NSR safety check on combined context
"""

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rag.retriever import retrieve, format_context
from nsr.rules import check
from agents.llm_provider import invoke_with_fallback, format_engine_notice


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
}


def _get_config(intent: str) -> dict:
    """Return config for given intent, default to general_chat."""
    return INTENT_CONFIG.get(intent, INTENT_CONFIG["general_chat"])


# ── Main Function ────────────────────────────────

def run(user_message: str,
        conversation_history: list = None,
        client_context: str = "",
        intent: str = "general_chat",
        model_key: str = None) -> dict:
    """
    Handle a conversational turn with the appropriate engine.

    Args:
        user_message:         Current user message
        conversation_history: List of {"role": str, "content": str}
        client_context:       Original client conversation (for follow-ups)
        intent:               Detected intent from classifier

    Returns:
        {
            "response":     str,        # with engine notice if fallback used
            "engine":       str,        # 'claude', 'openai', 'openai_fallback', etc.
            "intent":       str,
            "sections":     list,
            "nsr_warnings": list,
        }
    """
    cfg = _get_config(intent)

    # Build message chain
    messages = [SystemMessage(content=cfg["prompt"])]

    # Add knowledge base context if needed
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

    # Add current user message
    messages.append(HumanMessage(content=user_message))

    # Use frontend-chosen model if provided, otherwise use intent default
    chosen_model = model_key or cfg["default_model"]

    # Invoke with centralized fallback logic
    output, engine = invoke_with_fallback(
        messages,
        model_key=chosen_model,
        temperature=cfg["temperature"],
    )

    # Prepend engine notice if fallback was used
    notice       = format_engine_notice(engine)
    final_output = notice + output

    # NSR safety check on combined context
    combined_text = user_message + " " + output + " " + (client_context or "")
    violations    = check(combined_text)

    return {
        "response":     final_output,
        "engine":       engine,
        "intent":       intent,
        "sections": [
            {"id": "ai_response", "title": "AI Response", "content": final_output},
        ],
        "nsr_warnings": violations,
    }