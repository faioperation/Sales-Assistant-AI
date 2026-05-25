import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI (for chatbot / random discussion) ───
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL    = "gpt-4o"

# ── Anthropic (for sales analysis agents) ──────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL      = "claude-sonnet-4-5"

# Backward compatibility
LLM_MODEL = OPENAI_MODEL

# ── ChromaDB ───────────────────────────────────
CHROMA_API_KEY    = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT     = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE   = os.getenv("CHROMA_DATABASE")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "sales_assistant")

# ── RAG ────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K           = 5
# ── Agent Training API ─────────────────────────
TRAINING_API_URL = os.getenv("TRAINING_API_URL")