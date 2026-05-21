"""
Conversation Session Store
============================
In-memory storage for conversation history per session.
Backend can use session_id to maintain context across requests.

Replace with Redis/DB for production multi-server setup.
"""

from datetime import datetime
from typing import Optional


# In-memory session store
# Format: {session_id: {"history": [...], "created_at": str, "last_used": str}}
_SESSIONS: dict = {}

# Max messages to keep per session (avoid token bloat)
MAX_HISTORY = 20


def add_message(session_id: str, role: str, content: str) -> None:
    """
    Add a message to a session's history.
    role: 'user' or 'assistant'
    """
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = {
            "history":    [],
            "created_at": datetime.now().isoformat(),
            "last_used":  datetime.now().isoformat(),
        }

    _SESSIONS[session_id]["history"].append({
        "role":    role,
        "content": content,
    })
    _SESSIONS[session_id]["last_used"] = datetime.now().isoformat()

    # Trim if too long
    if len(_SESSIONS[session_id]["history"]) > MAX_HISTORY:
        _SESSIONS[session_id]["history"] = _SESSIONS[session_id]["history"][-MAX_HISTORY:]


def get_history(session_id: str) -> list:
    """Get history for a session. Returns empty list if not found."""
    if session_id not in _SESSIONS:
        return []
    return _SESSIONS[session_id]["history"]


def clear_session(session_id: str) -> bool:
    """Delete a session's history."""
    if session_id in _SESSIONS:
        del _SESSIONS[session_id]
        return True
    return False


def list_sessions() -> list:
    """List all active sessions (for debugging)."""
    return [
        {
            "session_id":     sid,
            "message_count":  len(data["history"]),
            "created_at":     data["created_at"],
            "last_used":      data["last_used"],
        }
        for sid, data in _SESSIONS.items()
    ]


def merge_history(
    session_id: Optional[str],
    frontend_history: Optional[list],
) -> list:
    """
    Merge backend session history with frontend-provided history.
    Frontend history takes priority if provided.
    """
    if frontend_history:
        return frontend_history
    if session_id:
        return get_history(session_id)
    return []