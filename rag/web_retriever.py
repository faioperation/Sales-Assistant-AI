import re
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import OPENAI_API_KEY


# ── Feature flag ─────────────────────────────────
# Web search is currently DISABLED.
# Set to True to re-enable the DuckDuckGo → fetch → OpenAI pipeline.
# When False, should_web_search() always returns False, so every
# caller falls through to its normal RAG flow. All pipeline code
# below stays intact and ready to use.
WEB_SEARCH_ENABLED = False


# ── LLM for analysis ────────────────────────────

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    temperature=0.3,
)


# ── Web search trigger keywords ─────────────────

WEB_SEARCH_TRIGGERS = [
    # English
    "search", "latest", "recent", "current", "find out",
    "look up", "what's new", "up to date", "up-to-date",
    "google", "browse", "check online", "web search",
    "stackoverflow", "medium", "reddit",
    "latest info", "latest news", "latest update",
    "real world", "real-world example", "find", "look for", "fetch", "get info", "know about"
    # Bangla / mixed
    "search করো", "search কর", "খুঁজে দাও", "খুঁজো",
    "latest জানাও", "নতুন কী", "আপডেট দাও",
    "অনলাইনে খোঁজো", "ইন্টারনেটে খোঁজো",
    "জানতে চাই", "find koro", "find করো", "latest bolo", "latest জানাও", "kisu bolo", "kisu janio"
]

# Target sources for DuckDuckGo
# Strict site targeting — DuckDuckGo site: operator
TARGET_SITES = {
    "stackoverflow": "site:stackoverflow.com",
    "medium":        "site:medium.com",
    "reddit":        "site:reddit.com",
}

# Allowed domains for result filtering
ALLOWED_DOMAINS = [
    "stackoverflow.com",
    "medium.com",
    "reddit.com",
]

ANALYSIS_SYSTEM_PROMPT = """
You are a senior technical and sales analyst. 
You have been given search results from StackOverflow, Medium, and Reddit.

Your job:
1. Extract the most relevant and accurate information
2. Summarize key insights in a clear, structured way
3. Connect the findings to the user's actual question
4. Mention the source type (StackOverflow/Medium/Reddit) where relevant
5. Be honest if the results are not directly relevant

Format:
- Start with a direct answer to the question
- Then provide supporting details from sources
- End with practical recommendations

Do NOT hallucinate. Only use what's in the provided content.
"""


# ── Trigger Detection ────────────────────────────

def should_web_search(text: str) -> bool:
    """Check if user message contains web search triggers."""
    if not WEB_SEARCH_ENABLED:
        return False
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in WEB_SEARCH_TRIGGERS)


# ── DuckDuckGo Search ────────────────────────────

def _get_source_type(url: str) -> str:
    """Map URL to source type name."""
    if "stackoverflow.com" in url:
        return "StackOverflow"
    elif "medium.com" in url:
        return "Medium"
    elif "reddit.com" in url:
        return "Reddit"
    else:
        return "Unknown"

def _is_allowed_url(url: str) -> bool:
    """Strictly allow only StackOverflow, Medium, Reddit URLs."""
    return any(domain in url for domain in ALLOWED_DOMAINS)


def _search_duckduckgo(query: str, max_results: int = 6) -> list[dict]:
    """
    Search DuckDuckGo strictly on StackOverflow, Medium, Reddit.
    Uses site: operator per site + filters results by domain.
    Returns list of {title, url, snippet}.
    """
    results = []

    with DDGS() as ddgs:
        for site_name, site_operator in TARGET_SITES.items():
            try:
                # Strictly target this site only
                site_query   = f"{query} {site_operator}"
                site_results = list(ddgs.text(
                    site_query,
                    max_results=2,
                    safesearch="off",
                ))

                # Extra filter — only keep URLs from allowed domains
                for r in site_results:
                    url = r.get("href", "")
                    if url and _is_allowed_url(url):
                        results.append(r)
                    else:
                        print(f"[WebRetriever] Skipping non-target URL: {url}")

            except Exception as e:
                print(f"[WebRetriever] DuckDuckGo search failed for {site_name}: {e}")
                continue

    # Deduplicate by URL
    seen = set()
    unique = []
    for r in results:
        url = r.get("href", "")
        if url and url not in seen:
            seen.add(url)
            unique.append({
                "title":   r.get("title", ""),
                "url":     url,
                "snippet": r.get("body", ""),
            })

    print(f"[WebRetriever] Found {len(unique)} results from target sites.")
    return unique[:max_results]


# ── Content Fetcher ──────────────────────────────

def _fetch_page_content(url: str, max_chars: int = 2000) -> str:
    """
    Fetch and extract text content from a URL.
    Returns empty string on failure.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer",
                         "header", "aside", "advertisement"]):
            tag.decompose()

        # Extract main content
        # StackOverflow → answers
        if "stackoverflow.com" in url:
            answers = soup.find_all("div", class_=re.compile(r"answer"))
            if answers:
                text = " ".join(a.get_text(separator=" ", strip=True)
                                for a in answers[:2])
                return text[:max_chars]

        # Medium → article body
        if "medium.com" in url:
            article = soup.find("article")
            if article:
                return article.get_text(separator=" ", strip=True)[:max_chars]

        # Reddit → post + top comments
        if "reddit.com" in url:
            posts = soup.find_all("div", class_=re.compile(r"Post|Comment"))
            if posts:
                text = " ".join(p.get_text(separator=" ", strip=True)
                                for p in posts[:3])
                return text[:max_chars]

        # Generic fallback
        main = soup.find("main") or soup.find("article") or soup.body
        if main:
            return main.get_text(separator=" ", strip=True)[:max_chars]

        return soup.get_text(separator=" ", strip=True)[:max_chars]

    except Exception as e:
        print(f"[WebRetriever] Failed to fetch {url}: {e}")
        return ""


# ── Build Context from Results ───────────────────

def _build_search_context(query: str, results: list[dict]) -> str:
    """
    Fetch content from each result and build a combined context string.
    """
    if not results:
        return "No search results found."

    sections = [f"=== WEB SEARCH RESULTS FOR: '{query}' ===\n"]

    for i, result in enumerate(results, 1):
        title   = result.get("title", "No title")
        url     = result.get("url", "")
        snippet = result.get("snippet", "")

        # Determine source type
        source = _get_source_type(url)

        # Fetch full content
        content = _fetch_page_content(url)
        body    = content if content else snippet

        sections.append(
            f"[{i}] {source} — {title}\n"
            f"URL: {url}\n"
            f"Content: {body}\n"
        )

    return "\n".join(sections)


# ── Main Public Function ─────────────────────────

def retrieve_and_analyze(
    user_query: str,
    extra_context: str = "",
) -> dict:
    """
    Full pipeline: search → fetch → analyze.

    Args:
        user_query:    The user's question/request
        extra_context: Additional context (RAG data, conversation history summary)

    Returns:
        {
            "answer":          str,   ← OpenAI analyzed response
            "sources":         list,  ← [{title, url, source_type}]
            "search_context":  str,   ← raw fetched content
            "web_search_used": bool,
        }
    """
    print(f"[WebRetriever] Searching for: {user_query}")

    # ── Step 1: Search ───────────────────────────
    results = _search_duckduckgo(user_query)

    if not results:
        return {
            "answer":          "Web search returned no results. Please try rephrasing.",
            "sources":         [],
            "search_context":  "",
            "web_search_used": True,
        }

    # ── Step 2: Fetch + build context ───────────
    search_context = _build_search_context(user_query, results)

    # ── Step 3: Analyze with OpenAI ─────────────
    user_content = f"USER QUESTION:\n{user_query}"

    if extra_context:
        user_content += f"\n\nADDITIONAL CONTEXT:\n{extra_context}"

    user_content += f"\n\n{search_context}"

    messages = [
        SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        response = llm.invoke(messages)
        answer   = response.content
    except Exception as e:
        print(f"[WebRetriever] OpenAI analysis failed: {e}")
        answer = f"Search completed but analysis failed: {e}\n\n{search_context}"

    # ── Step 4: Build sources list ───────────────
    sources = []
    for r in results:
        url = r.get("url", "")
        source_type = _get_source_type(url)

        sources.append({
            "title":       r.get("title", ""),
            "url":         url,
            "source_type": source_type,
        })

    print(f"[WebRetriever] Done. {len(sources)} sources found.")

    return {
        "answer":          answer,
        "sources":         sources,
        "search_context":  search_context,
        "web_search_used": True,
    }