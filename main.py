import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from orchestrator.full_analysis import run_full_analysis
from orchestrator.session_store import add_message, merge_history

app = FastAPI(title="Sales Assistant AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Model ──────────────────────────────

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


# ── Full Analysis (Single Entry Point) ─────────

@app.post("/full-analysis")
def full_analysis_endpoint(req: ChatRequest):
    """
    Single entry point for all AI requests.

    Response format (harmony):
    {
        "agent":                 str,
        "intent":                str,
        "mode":                  str,
        "page":                  str,
        "engine":                str,
        "sections": [
            {"id": str, "title": str, "content": str}
        ],
        "nsr_warnings":          list,
        "combined_nsr_warnings": list,
        "raw":                   str,
        "conversation_id":       str
    }
    """
    chat_id      = req.conversation_id
    history      = merge_history(chat_id, req.conversation_history)
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
            add_message(chat_id, "assistant", result.get("raw", ""))

    result["conversation_id"] = chat_id
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)