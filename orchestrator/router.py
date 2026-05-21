"""
Router — user input দেখে কোন agent-এ পাঠাবে সেটা decide করে।
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import OPENAI_API_KEY, LLM_MODEL

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0,
)

ROUTER_PROMPT = """
You are a routing agent. Read the user input and classify it into exactly ONE of these intents:

- sales_bot        : User wants to analyze a Fiverr client conversation or get help with a client reply
- service_guide    : User wants technical information about a service (how to build, challenges, limitations)
- quotation        : User wants to create a project proposal or price estimate
- alternative_guide: User has a problem — impossible requirement, budget mismatch, or wrong tech stack
- system_prompt    : General question, document review, image analysis, or anything else

Reply with ONLY the intent keyword. Nothing else.
Examples:
- "Here is a chat with my client, help me reply" → sales_bot
- "How does n8n automation work?" → service_guide
- "Create a quotation for a WordPress site" → quotation
- "Client wants real-time sync with Zapier, is this possible?" → alternative_guide
- "Review this proposal document" → system_prompt
"""


def detect_intent(user_input: str,
                  document_content: str = "",
                  image_description: str = "") -> str:
    """
    User input থেকে intent detect করে।

    Returns:
        str: one of [sales_bot, service_guide, quotation,
                     alternative_guide, system_prompt]
    """
    content = user_input
    if document_content:
        content += f"\n[Document attached: {document_content[:200]}...]"
    if image_description:
        content += f"\n[Image attached: {image_description[:100]}...]"

    messages = [
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=content),
    ]

    response = llm.invoke(messages)
    intent   = response.content.strip().lower()

    valid_intents = {
        "sales_bot", "service_guide", "quotation",
        "alternative_guide", "system_prompt",
    }

    return intent if intent in valid_intents else "system_prompt"


def route(state: dict) -> str:
    """LangGraph conditional edge-এর জন্য — intent return করে।"""
    return state.get("intent", "system_prompt")