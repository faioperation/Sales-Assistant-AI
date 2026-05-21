"""
Hybrid Intent Classifier
==========================
Two-stage classification for speed + accuracy:

Stage 1 - Fast rule-based detection (no LLM call):
  - Catches obvious cases instantly
  - Uses keywords, length, patterns

Stage 2 - LLM-based classifier (fallback):
  - Only when rules are uncertain
  - Uses gpt-4o-mini (fast + cheap)

Intent categories:
- sales_analysis  : Long client conversation, needs full 3-agent analysis
- sales_followup  : Follow-up about ongoing client situation
- tech_discussion : Technical/business question about tools/platforms
- general_chat    : Random general knowledge question
"""

import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import OPENAI_API_KEY


# ── Fast classifier LLM (only used as fallback) ──

classifier_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
    max_tokens=20,
)


# ── Keyword sets for rule-based detection ────────

# Strong signals that user pasted a client conversation
CLIENT_PASTE_PATTERNS = [
    r"\bhi[,\s]+i\s+need\b",
    r"\bhello[,\s]+i\s+(want|need|looking)",
    r"\bclient\s+(says|wants|asked|message)",
    r"\bwhat\s+is\s+your\s+(price|cost|rate|timeline)",
    r"\bcan\s+you\s+(build|develop|create|make)\s+",
    r"\bi\s+want\s+to\s+(build|develop|create)\s+",
    r"\bmy\s+(business|company|website|app|store)\s+",
    r"\@\w+",                              # @username mentions
    r"\bbudget\s+(is|of|around|under)\b",
    r"\b\$\d+",                            # dollar amounts
    r"https?://\S+",                       # URLs
]

# Follow-up indicators (user asking what to do next)
SALES_FOLLOWUP_KEYWORDS = [
    "what should i reply", "what should i say", "what should i ask",
    "how should i reply", "how should i respond", "what to say",
    "create quotation", "make quotation", "draft quotation",
    "draft a reply", "write a reply", "give me a reply",
    "create proposal", "make proposal",
    "ki bolbo", "ki uttor dibo", "ami ki",
    "this client", "the client", "for this project",
    "next step", "follow up", "follow-up",
    "negotiate", "negotiation",
    "should i accept", "should i decline",
]

# Tech discussion indicators
TECH_KEYWORDS = [
    "how does", "how do i", "how to use", "how to build", "how to implement",
    "what is the difference between",
    "should i use", "is it better to",
    "explain how", "explain the",
    "what are the limitations", "what are the features",
    "pricing of", "cost of using", "rate limit",
    "tech stack for", "best tool for", "best platform for",
    "n8n", "make.com", "zapier", "vapi", "voiceflow", "dialogflow",
    "langchain", "langgraph", "supabase", "firebase",
    "wordpress", "shopify", "wix", "webflow", "squarespace",
    "flutter", "react native", "kotlin", "swift",
]

# General chat indicators (NOT sales/tech related)
GENERAL_CHAT_KEYWORDS = [
    "what is smart contract", "what is blockchain", "what is bitcoin",
    "what is the weather", "tell me a joke", "tell me about",
    "who is", "when did", "where is", "history of",
    "meaning of", "definition of",
    "capital of", "currency of",
    "how do you feel", "are you ai", "what model",
]


# ── Helper functions ─────────────────────────────

def _matches_any_pattern(text: str, patterns: list) -> int:
    """Count how many regex patterns match."""
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))


def _contains_any(text: str, keywords: list) -> int:
    """Count how many keywords appear in text."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


# ── Stage 1: Rule-based detection ─────────────────

def _rule_based_classify(text: str, has_history: bool) -> str | None:
    """
    Try to classify using rules. Returns intent or None if uncertain.
    """
    text_lower = text.lower().strip()
    word_count = len(text.split())

    # Strong signal: client conversation paste (long + multiple patterns)
    client_paste_score = _matches_any_pattern(text, CLIENT_PASTE_PATTERNS)
    if word_count > 40 and client_paste_score >= 2:
        return "sales_analysis"

    # Strong signal: follow-up phrases
    followup_score = _contains_any(text_lower, SALES_FOLLOWUP_KEYWORDS)
    if followup_score >= 1 and has_history:
        return "sales_followup"
    if followup_score >= 2:
        return "sales_followup"

    # Strong signal: tech discussion
    tech_score = _contains_any(text_lower, TECH_KEYWORDS)
    if tech_score >= 2:
        return "tech_discussion"

    # Strong signal: general chat
    general_score = _contains_any(text_lower, GENERAL_CHAT_KEYWORDS)
    if general_score >= 1:
        return "general_chat"

    # Very short message with history -> follow-up
    if has_history and word_count < 20:
        return "sales_followup"

    # Uncertain — fall back to LLM
    return None


# ── Stage 2: LLM-based fallback ──────────────────

CLASSIFIER_PROMPT = """You are an intent classifier. Classify the user message into exactly ONE intent:

- sales_analysis: User pasted a complete client message/conversation (50+ words with client requirements, budget, project details)
- sales_followup: User asking about an ongoing client situation, wants help with reply/quotation/strategy
- tech_discussion: Technical question about tools/platforms/frameworks for work (n8n, Vapi, WordPress, AI agents, etc.)
- general_chat: Random general knowledge question NOT about sales/tech work (smart contract, blockchain, weather, history)

Reply with ONLY the intent name. No punctuation, no explanation."""


def _llm_classify(text: str, has_history: bool) -> str:
    """Use LLM for ambiguous cases."""
    history_hint = "\n[Note: Conversation history exists]" if has_history else ""

    messages = [
        SystemMessage(content=CLASSIFIER_PROMPT),
        HumanMessage(content=f"Message: {text}{history_hint}"),
    ]

    try:
        response = classifier_llm.invoke(messages)
        intent   = response.content.strip().lower().replace(".", "").replace(",", "")

        valid_intents = {
            "sales_analysis", "sales_followup",
            "tech_discussion", "general_chat",
        }

        for valid in valid_intents:
            if valid in intent:
                return valid

        return "general_chat"

    except Exception as e:
        print(f"[Classifier] LLM error, using fallback: {e}")
        return "sales_followup" if has_history else "general_chat"


# ── Main Entry ───────────────────────────────────

def classify(user_message: str, has_history: bool = False) -> str:
    """
    Hybrid classification: rules first, LLM fallback.

    Args:
        user_message: The current user message
        has_history:  Whether previous conversation exists

    Returns:
        One of: 'sales_analysis', 'sales_followup',
                'tech_discussion', 'general_chat'
    """
    # Stage 1: Try rule-based
    rule_result = _rule_based_classify(user_message, has_history)
    if rule_result:
        return rule_result

    # Stage 2: Fall back to LLM
    return _llm_classify(user_message, has_history)


# ── Debug helper ─────────────────────────────────

def classify_with_reason(user_message: str, has_history: bool = False) -> dict:
    """
    Returns intent + which stage decided it (for debugging).
    """
    rule_result = _rule_based_classify(user_message, has_history)

    if rule_result:
        return {
            "intent":      rule_result,
            "stage":       "rule_based",
            "llm_called":  False,
        }

    llm_result = _llm_classify(user_message, has_history)
    return {
        "intent":      llm_result,
        "stage":       "llm_fallback",
        "llm_called":  True,
    }