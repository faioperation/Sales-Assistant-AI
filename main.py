import base64
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
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


# ── Supported image MIME types ─────────────────

SUPPORTED_IMAGE_TYPES = {
    "image/png":  "image/png",
    "image/jpeg": "image/jpeg",
    "image/jpg":  "image/jpeg",
    "image/webp": "image/webp",
    "image/gif":  "image/gif",
}


# ── Request Model (JSON endpoint) ─────────────

class ChatRequest(BaseModel):
    user_input:           str
    document_content:     Optional[str] = ""
    image_description:    Optional[str] = ""
    image_data:           Optional[str] = ""   # base64 or data:image URL or http URL
    conversation_history: Optional[List[dict]] = []
    conversation_id:      Optional[str] = None
    model_key:            Optional[str] = None
    page_context:         Optional[str] = None


# ── Health ─────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "Sales Assistant AI"}


# ── Full Analysis — JSON (existing endpoint) ───

@app.post("/full-analysis")
def full_analysis_endpoint(req: ChatRequest):
    """
    JSON endpoint. image_data accepts:
    - base64 string
    - data:image/png;base64,... 
    - https://... URL
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
        image_data=req.image_data or "",
    )

    if chat_id:
        add_message(chat_id, "user", req.user_input)
        if result.get("mode") == "conversation":
            add_message(chat_id, "assistant", result.get("response", ""))
        else:
            add_message(chat_id, "assistant", result.get("raw", ""))

    result["conversation_id"] = chat_id
    return result


# ── Full Analysis — File Upload (new endpoint) ─

@app.post("/full-analysis-image")
async def full_analysis_image_endpoint(
    image:           UploadFile         = File(...),
    user_input:      str                = Form(...),
    page_context:    Optional[str]      = Form(None),
    model_key:       Optional[str]      = Form(None),
    conversation_id: Optional[str]      = Form(None),
):
    """
    Multipart/form-data endpoint for file upload.

    Postman setup:
    - POST /full-analysis-image
    - Body → form-data
    - image       → File  → select your image file
    - user_input  → Text  → your question/instruction
    - page_context → Text → sales_bot / service_guide / alternative_guide (optional)
    - model_key   → Text  → claude-sonnet / gpt-4o (optional)
    - conversation_id → Text → session id (optional)
    """

    # ── Validate image type ──────────────────────
    content_type = image.content_type or ""
    mime_type    = SUPPORTED_IMAGE_TYPES.get(content_type)

    if not mime_type:
        return {
            "error":   f"Unsupported image type: {content_type}",
            "allowed": list(SUPPORTED_IMAGE_TYPES.keys()),
        }

    # ── Read file and convert to base64 ─────────
    image_bytes  = await image.read()

    if not image_bytes:
        return {"error": "Empty image file received"}

    b64_string = base64.b64encode(image_bytes).decode("utf-8")
    image_data = f"data:{mime_type};base64,{b64_string}"

    # ── Run full analysis ────────────────────────
    chat_id      = conversation_id
    history      = merge_history(chat_id, [])
    is_follow_up = bool(history)

    result = run_full_analysis(
        conversation_text=user_input,
        conversation_history=history,
        is_follow_up=is_follow_up,
        model_key=model_key,
        page_context=page_context,
        image_data=image_data,
    )

    # ── Save to session ──────────────────────────
    if chat_id:
        add_message(chat_id, "user", user_input)
        if result.get("mode") == "conversation":
            add_message(chat_id, "assistant", result.get("response", ""))
        else:
            add_message(chat_id, "assistant", result.get("raw", ""))

    result["conversation_id"] = chat_id
    result["image_filename"]  = image.filename
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8036, reload=False)