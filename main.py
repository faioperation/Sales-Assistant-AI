import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from orchestrator.full_analysis import run_full_analysis, _is_quotation_request
from orchestrator.session_store import (
    add_message, merge_history,
)
from agents import (
    sales_bot, service_guide, alternative_guide, conversation,
)

app = FastAPI(title="Sales Assistant AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request model ──────────────────────────────

class ChatRequest(BaseModel):
    user_input:           str
    document_content:     Optional[str] = ""
    image_description:    Optional[str] = ""
    conversation_history: Optional[List[dict]] = []
    conversation_id:      Optional[str] = None
    model_key:            Optional[str] = None
    page_context:         Optional[str] = None


# ── Health ─────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "Sales Assistant AI"}


# ── Fiverr Sales Bot ───────────────────────────

@app.post("/sales-bot")
def sales_bot_endpoint(req: ChatRequest):
    """
    Fiverr Sales Bot endpoint.
    - First message: Runs sales_bot agent (full client analysis)
    - Follow-up: Conversational mode with client context
    """
    chat_id = req.conversation_id
    history = merge_history(chat_id, req.conversation_history)

    if not history:
        result = sales_bot.run(
            conversation=req.user_input,
            model_key=req.model_key or "claude-sonnet",
        )
        mode = "analysis"
        intent = "sales_analysis"
    else:
        client_context = ""
        for msg in history:
            if msg.get("role") == "user":
                client_context = msg.get("content", "")
                break

        result = conversation.run(
            user_message=req.user_input,
            conversation_history=history,
            client_context=client_context,
            intent="sales_followup",
            model_key=req.model_key,
        )
        mode = "conversation"
        intent = "sales_followup"

    if chat_id:
        add_message(chat_id, "user", req.user_input)
        if mode == "conversation":
            add_message(chat_id, "assistant", result.get("response", ""))
        else:
            add_message(chat_id, "assistant", "[Sales Bot analysis completed]")

    return {
        "agent":           "sales_bot",
        "intent":          intent,
        "mode":            mode,
        "page":            "sales_bot",
        "conversation_id": chat_id,
        **result,
    }


# ── Service Guide (with auto Quotation) ────────

@app.post("/service-guide")
def service_guide_endpoint(req: ChatRequest):
    """
    Service Guide endpoint.
    - Quotation keywords detected: Runs quotation generator
    - First service question: Runs service_guide agent
    - Follow-up: Conversational with tech context
    """
    chat_id = req.conversation_id
    history = merge_history(chat_id, req.conversation_history)

    client_context = ""
    if history:
        for msg in history:
            if msg.get("role") == "user":
                client_context = msg.get("content", "")
                break

    if _is_quotation_request(req.user_input):
        service_desc = client_context if client_context else req.user_input
        result = service_guide.quotation(
            service_description=service_desc,
            model_key=req.model_key or "claude-sonnet",
        )
        agent_name = "quotation"
        intent = "quotation_generation"
        mode = "quotation"
    elif not history:
        result = service_guide.guide(
            service_description=req.user_input,
            model_key=req.model_key or "claude-sonnet",
        )
        agent_name = "service_guide"
        intent = "service_guide"
        mode = "guide"
    else:
        result = conversation.run(
            user_message=req.user_input,
            conversation_history=history,
            client_context=client_context,
            intent="tech_discussion",
            model_key=req.model_key,
        )
        agent_name = "service_guide"
        intent = "tech_discussion"
        mode = "conversation"

    if chat_id:
        add_message(chat_id, "user", req.user_input)
        if mode == "conversation":
            add_message(chat_id, "assistant", result.get("response", ""))
        else:
            add_message(chat_id, "assistant", f"[{agent_name} completed]")

    return {
        "agent":           agent_name,
        "intent":          intent,
        "mode":            mode,
        "page":            "service_guide",
        "conversation_id": chat_id,
        **result,
    }


# ── Alternative Guide ──────────────────────────

@app.post("/alternative-guide")
def alternative_guide_endpoint(req: ChatRequest):
    """
    Alternative Guide endpoint.
    - First message: Runs alternative_guide agent
    - Follow-up: Conversational with context
    """
    chat_id = req.conversation_id
    history = merge_history(chat_id, req.conversation_history)

    client_context = ""
    if history:
        for msg in history:
            if msg.get("role") == "user":
                client_context = msg.get("content", "")
                break

    if not history:
        result = alternative_guide.run(
            problem_description=req.user_input,
            model_key=req.model_key or "claude-sonnet",
        )
        intent = "alternative_guide"
        mode = "alternative"
    else:
        result = conversation.run(
            user_message=req.user_input,
            conversation_history=history,
            client_context=client_context,
            intent="sales_followup",
            model_key=req.model_key,
        )
        intent = "sales_followup"
        mode = "conversation"

    if chat_id:
        add_message(chat_id, "user", req.user_input)
        if mode == "conversation":
            add_message(chat_id, "assistant", result.get("response", ""))
        else:
            add_message(chat_id, "assistant", "[Alternative Guide completed]")

    return {
        "agent":           "alternative_guide",
        "intent":          intent,
        "mode":            mode,
        "page":            "alternative_guide",
        "conversation_id": chat_id,
        **result,
    }


# ── Full Analysis (System Prompt page) ─────────

@app.post("/full-analysis")
def full_analysis_endpoint(req: ChatRequest):
    """
    System Prompt page endpoint.
    Multi-agent or single conversational response based on intent.
    """
    chat_id = req.conversation_id
    history = merge_history(chat_id, req.conversation_history)
    is_follow_up = bool(history)

    result = run_full_analysis(
        conversation_text=req.user_input,
        conversation_history=history,
        is_follow_up=is_follow_up,
        model_key=req.model_key,
        page_context=req.page_context,
    )

    if chat_id:
        add_message(chat_id, "user", req.user_input)
        if result.get("mode") == "conversation":
            add_message(chat_id, "assistant", result.get("response", ""))
        else:
            add_message(chat_id, "assistant", "[Full analysis completed]")

    result["conversation_id"] = chat_id
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)