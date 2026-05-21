"""
Shared state — LangGraph-এর সব nodes এই state share করে।
"""

from typing import TypedDict, Optional


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────
    user_input:           str
    document_content:     Optional[str]
    image_description:    Optional[str]
    conversation_history: Optional[list]

    # ── Routing ────────────────────────────────
    intent:               Optional[str]   # sales_bot | service_guide |
                                          # quotation | alternative_guide |
                                          # system_prompt

    # ── Agent outputs ──────────────────────────
    agent_response:       Optional[dict]

    # ── Final ──────────────────────────────────
    final_output:         Optional[dict]
    error:                Optional[str]