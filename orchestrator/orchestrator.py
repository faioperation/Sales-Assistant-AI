from langgraph.graph import StateGraph, END

from orchestrator.state  import AgentState
from orchestrator.router import detect_intent, route

from agents import sales_bot
from agents import service_guide
from agents import alternative_guide
from agents import system_prompt as system_prompt_agent
from concurrent.futures import ThreadPoolExecutor
from agents import sales_bot, service_guide, alternative_guide

# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_router(state: AgentState) -> AgentState:
    """Intent detect করে state-এ set করে।"""
    intent = detect_intent(
        user_input=state["user_input"],
        document_content=state.get("document_content", ""),
        image_description=state.get("image_description", ""),
    )
    return {**state, "intent": intent}


def node_sales_bot(state: AgentState) -> AgentState:
    """Fiverr Sales Bot agent run করে।"""
    result = sales_bot.run(
        conversation=state["user_input"],
    )
    return {
        **state,
        "agent_response": result,
        "final_output": {
            "agent":  "sales_bot",
            "intent": state["intent"],
            **result,
        },
    }


def node_service_guide(state: AgentState) -> AgentState:
    """Service Guide agent run করে।"""
    result = service_guide.guide(
        service_description=state["user_input"],
    )
    return {
        **state,
        "agent_response": result,
        "final_output": {
            "agent":  "service_guide",
            "intent": state["intent"],
            **result,
        },
    }


def node_quotation(state: AgentState) -> AgentState:
    """Quotation Generator run করে।"""
    result = service_guide.quotation(
        service_description=state["user_input"],
    )
    return {
        **state,
        "agent_response": result,
        "final_output": {
            "agent":  "quotation",
            "intent": state["intent"],
            **result,
        },
    }


def node_alternative_guide(state: AgentState) -> AgentState:
    """Alternative Guide agent run করে।"""
    result = alternative_guide.run(
        problem_description=state["user_input"],
    )
    return {
        **state,
        "agent_response": result,
        "final_output": {
            "agent":  "alternative_guide",
            "intent": state["intent"],
            **result,
        },
    }


def node_system_prompt(state: AgentState) -> AgentState:
    """System Prompt (free-form) agent run করে।"""
    result = system_prompt_agent.run(
        user_input=state["user_input"],
        document_content=state.get("document_content", ""),
        image_description=state.get("image_description", ""),
        conversation_history=state.get("conversation_history", []),
    )
    return {
        **state,
        "agent_response": result,
        "final_output": {
            "agent":  "system_prompt",
            "intent": state["intent"],
            **result,
        },
    }


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)


    graph.add_node("router",           node_router)
    graph.add_node("sales_bot",        node_sales_bot)
    graph.add_node("service_guide",    node_service_guide)
    graph.add_node("quotation",        node_quotation)
    graph.add_node("alternative_guide",node_alternative_guide)
    graph.add_node("system_prompt",    node_system_prompt)

    # Entry point
    graph.set_entry_point("router")


    graph.add_conditional_edges(
        "router",
        route,
        {
            "sales_bot":         "sales_bot",
            "service_guide":     "service_guide",
            "quotation":         "quotation",
            "alternative_guide": "alternative_guide",
            "system_prompt":     "system_prompt",
        },
    )


    for node in ["sales_bot", "service_guide", "quotation",
                 "alternative_guide", "system_prompt"]:
        graph.add_edge(node, END)

    return graph.compile()



_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph




def run(user_input: str,
        document_content: str = "",
        image_description: str = "",
        conversation_history: list = None) -> dict:
    """
    Frontend থেকে আসা request handle করে।

    Args:
        user_input:           user-এর message/question
        document_content:     uploaded document text (optional)
        image_description:    image description (optional)
        conversation_history: chat history (optional)

    Returns:
        {
            "agent":  str,     ← কোন agent handle করলো
            "intent": str,     ← detected intent
            ...                ← agent-specific response fields
        }
    """
    graph = get_graph()

    initial_state: AgentState = {
        "user_input":           user_input,
        "document_content":     document_content,
        "image_description":    image_description,
        "conversation_history": conversation_history or [],
        "intent":               None,
        "agent_response":       None,
        "final_output":         None,
        "error":                None,
    }

    try:
        result = graph.invoke(initial_state)
        return result.get("final_output", {"error": "No output generated"})
    except Exception as e:
        return {"error": str(e), "agent": "orchestrator"}
    


    