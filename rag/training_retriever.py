import io
import time
import requests
import pypdf
from config import TRAINING_API_URL

CACHE_TTL = 60  # seconds — refresh training data every 60s

_cache: dict = {
    "data":       None,
    "fetched_at": 0.0,
}


# ── API Fetch ────────────────────────────────────

def _fetch_training_entries() -> list[dict]:
    """
    Fetch all training entries from backend API.
    Returns list of raw entry dicts.
    """
    try:
        resp = requests.get(TRAINING_API_URL, timeout=10)
        resp.raise_for_status()
        body = resp.json()
        if body.get("success") and isinstance(body.get("data"), list):
            return body["data"]
        return []
    except Exception as e:
        print(f"[TrainingRetriever] API fetch failed: {e}")
        return []


# ── PDF Extraction ───────────────────────────────

def _extract_pdf_text(url: str) -> str:
    """
    Download a PDF from URL and extract all text.
    Returns empty string on any failure.
    """
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        pdf_bytes = io.BytesIO(resp.content)
        reader    = pypdf.PdfReader(pdf_bytes)

        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())

        return "\n\n".join(pages)

    except Exception as e:
        print(f"[TrainingRetriever] PDF extraction failed for {url}: {e}")
        return ""


# ── Cache Layer ──────────────────────────────────

def _get_cached_entries() -> list[dict]:
    """
    Return cached training entries, refreshing if TTL expired.
    """
    now = time.time()
    if _cache["data"] is None or (now - _cache["fetched_at"]) > CACHE_TTL:
        print("[TrainingRetriever] Refreshing training data from API...")
        _cache["data"]       = _fetch_training_entries()
        _cache["fetched_at"] = now
    return _cache["data"]


# ── Main Public Function ─────────────────────────

def get_training_context() -> str:
    """
    Build a combined training context string from all training entries.

    For each entry:
    - Includes the prompt (behavior directive)
    - Fetches and includes PDF text if documentUrl is present

    Returns a formatted string ready to inject into agent system prompts.
    """
    entries = _get_cached_entries()
    if not entries:
        return ""

    sections = []

    for i, entry in enumerate(entries, 1):
        parts = []

        # --- Prompt / directive ---
        prompt = (entry.get("prompt") or "").strip()
        if prompt:
            parts.append(f"DIRECTIVE:\n{prompt}")

        # --- PDF document ---
        doc_url = (entry.get("documentUrl") or "").strip()
        if doc_url:
            pdf_text = _extract_pdf_text(doc_url)
            if pdf_text:
                parts.append(f"DOCUMENT CONTENT:\n{pdf_text[:3000]}")

        if parts:
            block = f"--- TRAINING ENTRY {i} ---\n" + "\n\n".join(parts)
            sections.append(block)

    if not sections:
        return ""

    return (
        "=== ADMIN TRAINING DATA ===\n"
        "The following directives and documents were provided by the admin.\n"
        "Apply this knowledge alongside your existing knowledge base.\n\n"
        + "\n\n".join(sections)
        + "\n=== END OF TRAINING DATA ==="
    )


def invalidate_cache() -> None:
    """Force refresh on next call (call after new training data is uploaded)."""
    _cache["data"]       = None
    _cache["fetched_at"] = 0.0