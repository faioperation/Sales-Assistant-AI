"""
NSR - Neurosymbolic Reasoning Rules
=====================================
Comprehensive symbolic rules covering all major platforms,
technologies, and risk categories for sales conversations.

Categories:
- BUDGET rules (per platform/service)
- TIMELINE rules (per platform/service)
- TECHNICAL_LIMITATION rules (real-world constraints)
- LEGAL_RISK / ETHICAL_RISK rules
- COMMUNICATION_RISK rules
- MISSING_INFO rules
- SCOPE_RISK rules
- HALLUCINATED_API rules
"""

import re


# ── Helper functions ─────────────────────────────────────────

def _has_budget_under(text: str, threshold: int) -> bool:
    """Detect any dollar amount under threshold."""
    matches = re.findall(r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)', text)
    for m in matches:
        amount = float(m.replace(",", ""))
        if 0 < amount < threshold:
            return True
    # Also check "X USD" pattern
    usd_matches = re.findall(r'(\d+(?:,\d{3})*)\s*(?:usd|dollar)', text.lower())
    for m in usd_matches:
        amount = float(m.replace(",", ""))
        if 0 < amount < threshold:
            return True
    return False


def _has_timeline_under(text: str, days_threshold: int) -> bool:
    """Detect timeline mentions shorter than threshold (in days)."""
    text_lower = text.lower()
    # Match X day(s)
    day_matches = re.findall(r'(\d+)\s*day', text_lower)
    for m in day_matches:
        if int(m) < days_threshold:
            return True
    # Match X week(s) -> convert to days
    week_matches = re.findall(r'(\d+)\s*week', text_lower)
    for m in week_matches:
        if int(m) * 7 < days_threshold:
            return True
    # Match X hour(s) -> any hour mention is unrealistic
    if re.search(r'\d+\s*hour', text_lower):
        return True
    return False


def _has_budget_mentioned(text: str) -> bool:
    """Check if any budget was mentioned."""
    return bool(re.search(
        r'\$\s*\d+|\d+\s*(?:usd|dollar|eur|inr|gbp)',
        text.lower()
    ))


def _has_timeline_mentioned(text: str) -> bool:
    """Check if any timeline was mentioned."""
    return bool(re.search(
        r'\b\d+\s*(?:day|week|month|hour)s?\b|\bdeadline\b|\basap\b',
        text.lower()
    ))


def _contains_any(text: str, keywords: list) -> bool:
    """Helper - true if any keyword in text."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


# ── Rule Definitions ─────────────────────────────────────────

RULES = [

    # ═══════════════════════════════════════════════════════════
    # BUDGET RULES - per platform/service category
    # ═══════════════════════════════════════════════════════════

    {
        "id": "voice_ai_low_budget",
        "keywords": ["voice", "vapi", "calling system", "ai call",
                     "outbound call", "inbound call", "telephony"],
        "checks": [{
            "trigger": lambda txt: _has_budget_under(txt, 800),
            "flag":    "LOW_BUDGET",
            "message": "Voice AI systems realistically start at $800+. Telephony integration, compliance, and testing require this minimum.",
        }],
    },

    {
        "id": "fullstack_low_budget",
        "keywords": ["full stack", "fullstack", "web app", "web application",
                     "react app", "next.js app", "saas"],
        "checks": [{
            "trigger": lambda txt: _has_budget_under(txt, 500),
            "flag":    "LOW_BUDGET",
            "message": "Full-stack web apps realistically start at $500+ for MVP. Complex apps with authentication, DB, and payments need $2000+.",
        }],
    },

    {
        "id": "ai_agent_low_budget",
        "keywords": ["ai agent", "llm agent", "autonomous agent", "agentic",
                     "ai assistant", "intelligent agent", "langchain", "langgraph"],
        "checks": [{
            "trigger": lambda txt: _has_budget_under(txt, 600),
            "flag":    "LOW_BUDGET",
            "message": "AI agent development starts at $600+. Multi-agent systems with RAG and tool-use need $2000+.",
        }],
    },

    {
        "id": "chatbot_low_budget",
        "keywords": ["chatbot", "chat bot", "voiceflow", "dialogflow",
                     "conversational ai", "botpress", "manychat"],
        "checks": [{
            "trigger": lambda txt: _has_budget_under(txt, 400),
            "flag":    "LOW_BUDGET",
            "message": "Custom chatbots (Voiceflow/Dialogflow + LLM) start at $400+. Custom matching logic and knowledge base integration add cost.",
        }],
    },

    {
        "id": "automation_low_budget",
        "keywords": ["n8n", "make.com", "zapier", "automation", "workflow",
                     "integromat"],
        "checks": [{
            "trigger": lambda txt: _has_budget_under(txt, 300),
            "flag":    "LOW_BUDGET",
            "message": "Automation workflows (n8n/Make/Zapier) start at $300+ for a 3-5 step flow. Complex multi-app integrations need $800+.",
        }],
    },

    {
        "id": "wordpress_low_budget",
        "keywords": ["wordpress", "wp", "elementor", "divi", "woocommerce"],
        "checks": [{
            "trigger": lambda txt: _has_budget_under(txt, 300),
            "flag":    "LOW_BUDGET",
            "message": "WordPress sites (theme + plugins + setup) start at $300+. Custom design or WooCommerce stores need $800-$2000+.",
        }],
    },

    {
        "id": "shopify_low_budget",
        "keywords": ["shopify"],
        "checks": [{
            "trigger": lambda txt: _has_budget_under(txt, 500),
            "flag":    "LOW_BUDGET",
            "message": "Production-ready Shopify store starts at $500+. Custom Liquid development or headless Shopify needs $2000+.",
        }],
    },

    {
        "id": "wix_low_budget",
        "keywords": ["wix", "velo"],
        "checks": [{
            "trigger": lambda txt: _has_budget_under(txt, 200),
            "flag":    "LOW_BUDGET",
            "message": "Professional Wix setup with Velo and custom design starts at $200+. Complex Wix Velo apps need $500+.",
        }],
    },

    {
        "id": "squarespace_low_budget",
        "keywords": ["squarespace"],
        "checks": [{
            "trigger": lambda txt: _has_budget_under(txt, 200),
            "flag":    "LOW_BUDGET",
            "message": "Squarespace setup with custom CSS/JS starts at $200+. E-commerce configuration adds $300-$500.",
        }],
    },

    {
        "id": "webflow_low_budget",
        "keywords": ["webflow"],
        "checks": [{
            "trigger": lambda txt: _has_budget_under(txt, 600),
            "flag":    "LOW_BUDGET",
            "message": "Custom Webflow build (responsive design, CMS, interactions) starts at $600+. With Memberstack/Wized integration needs $1500+.",
        }],
    },

    {
        "id": "flutter_low_budget",
        "keywords": ["flutter", "dart app"],
        "checks": [{
            "trigger": lambda txt: _has_budget_under(txt, 1500),
            "flag":    "LOW_BUDGET",
            "message": "Flutter cross-platform apps start at $1500+ for MVP. Complex apps with backend integration need $3000+.",
        }],
    },

    {
        "id": "react_native_low_budget",
        "keywords": ["react native", "expo"],
        "checks": [{
            "trigger": lambda txt: _has_budget_under(txt, 1500),
            "flag":    "LOW_BUDGET",
            "message": "React Native apps start at $1500+ for MVP. App Store submission, push notifications, and native modules add cost.",
        }],
    },

    {
        "id": "android_native_low_budget",
        "keywords": ["android app", "kotlin", "android native"],
        "checks": [{
            "trigger": lambda txt: _has_budget_under(txt, 1500),
            "flag":    "LOW_BUDGET",
            "message": "Native Android development starts at $1500+. Material Design, Play Store submission, and testing on multiple devices add cost.",
        }],
    },

    {
        "id": "ios_native_low_budget",
        "keywords": ["ios app", "swift", "ios native", "iphone app"],
        "checks": [{
            "trigger": lambda txt: _has_budget_under(txt, 2000),
            "flag":    "LOW_BUDGET",
            "message": "Native iOS development starts at $2000+. Apple Developer account ($99/yr), App Store review, and testing add cost.",
        }],
    },

    {
        "id": "ecommerce_custom_low_budget",
        "keywords": ["custom ecommerce", "custom e-commerce", "marketplace",
                     "multi-vendor"],
        "checks": [{
            "trigger": lambda txt: _has_budget_under(txt, 2000),
            "flag":    "LOW_BUDGET",
            "message": "Custom e-commerce or marketplace platforms start at $2000+. Payment integration, inventory, and orders need substantial work.",
        }],
    },

    # ═══════════════════════════════════════════════════════════
    # TIMELINE RULES - per platform/service category
    # ═══════════════════════════════════════════════════════════

    {
        "id": "voice_ai_unrealistic_timeline",
        "keywords": ["voice", "vapi", "ai call", "outbound call", "telephony"],
        "checks": [{
            "trigger": lambda txt: _has_timeline_under(txt, 14),
            "flag":    "UNREALISTIC_TIMELINE",
            "message": "Voice AI needs 14-20 days minimum for setup, testing, compliance checks, and telephony integration.",
        }],
    },

    {
        "id": "fullstack_unrealistic_timeline",
        "keywords": ["full stack", "fullstack", "web app", "saas"],
        "checks": [{
            "trigger": lambda txt: _has_timeline_under(txt, 21),
            "flag":    "UNREALISTIC_TIMELINE",
            "message": "Full-stack web apps need 30-45 days minimum for design, development, testing, and deployment.",
        }],
    },

    {
        "id": "ai_agent_unrealistic_timeline",
        "keywords": ["ai agent", "agentic", "llm agent", "ai assistant"],
        "checks": [{
            "trigger": lambda txt: _has_timeline_under(txt, 10),
            "flag":    "UNREALISTIC_TIMELINE",
            "message": "AI agent development needs 10-21 days minimum for prompt engineering, RAG setup, and testing.",
        }],
    },

    {
        "id": "chatbot_unrealistic_timeline",
        "keywords": ["chatbot", "voiceflow", "dialogflow", "conversational ai"],
        "checks": [{
            "trigger": lambda txt: _has_timeline_under(txt, 7),
            "flag":    "UNREALISTIC_TIMELINE",
            "message": "Even simple chatbots need 7-14 days for conversation design, prompt engineering, and testing.",
        }],
    },

    {
        "id": "wordpress_unrealistic_timeline",
        "keywords": ["wordpress", "wp", "woocommerce"],
        "checks": [{
            "trigger": lambda txt: _has_timeline_under(txt, 7),
            "flag":    "UNREALISTIC_TIMELINE",
            "message": "Proper WordPress sites need 7-14 days minimum. WooCommerce stores need 21+ days.",
        }],
    },

    {
        "id": "shopify_unrealistic_timeline",
        "keywords": ["shopify"],
        "checks": [{
            "trigger": lambda txt: _has_timeline_under(txt, 10),
            "flag":    "UNREALISTIC_TIMELINE",
            "message": "Shopify stores need 10-14 days minimum for theme setup, product import, payment config, and testing.",
        }],
    },

    {
        "id": "webflow_unrealistic_timeline",
        "keywords": ["webflow"],
        "checks": [{
            "trigger": lambda txt: _has_timeline_under(txt, 14),
            "flag":    "UNREALISTIC_TIMELINE",
            "message": "Custom Webflow sites need 14-21 days for design, responsive build, CMS setup, and QA.",
        }],
    },

    {
        "id": "flutter_unrealistic_timeline",
        "keywords": ["flutter", "react native", "mobile app"],
        "checks": [{
            "trigger": lambda txt: _has_timeline_under(txt, 30),
            "flag":    "UNREALISTIC_TIMELINE",
            "message": "Cross-platform mobile apps need 30-60 days minimum. App Store/Play Store submission adds 7-14 days.",
        }],
    },

    {
        "id": "android_unrealistic_timeline",
        "keywords": ["android app", "kotlin", "native android"],
        "checks": [{
            "trigger": lambda txt: _has_timeline_under(txt, 30),
            "flag":    "UNREALISTIC_TIMELINE",
            "message": "Native Android apps need 30-60 days minimum. Multi-device testing and Play Store submission add time.",
        }],
    },

    {
        "id": "ios_unrealistic_timeline",
        "keywords": ["ios app", "swift", "iphone app"],
        "checks": [{
            "trigger": lambda txt: _has_timeline_under(txt, 30),
            "flag":    "UNREALISTIC_TIMELINE",
            "message": "Native iOS apps need 30-60 days. Apple's review process adds 1-7 days. Plan accordingly.",
        }],
    },

    {
        "id": "automation_unrealistic_timeline",
        "keywords": ["n8n", "make.com", "zapier", "automation"],
        "checks": [{
            "trigger": lambda txt: _has_timeline_under(txt, 3),
            "flag":    "UNREALISTIC_TIMELINE",
            "message": "Even simple automations need 3-7 days for setup, integration auth, edge case handling, and testing.",
        }],
    },

    # ═══════════════════════════════════════════════════════════
    # TECHNICAL LIMITATION RULES
    # ═══════════════════════════════════════════════════════════

    {
        "id": "n8n_enterprise_scale",
        "keywords": ["n8n"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "million", "millions", "100k", "enterprise scale",
                "high volume", "1,000,000", "millions of"
            ]),
            "flag":    "TECHNICAL_LIMITATION",
            "message": "n8n has memory/performance limits at enterprise scale. For millions of records, use custom backend or cloud-native pipelines.",
        }],
    },

    {
        "id": "zapier_realtime_limit",
        "keywords": ["zapier"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "real-time", "realtime", "instant", "live sync",
                "live update", "milliseconds", "sub-second"
            ]),
            "flag":    "TECHNICAL_LIMITATION",
            "message": "Zapier is polling-based (1-15 min delay minimum). Cannot do true real-time sync. Use webhooks + custom backend instead.",
        }],
    },

    {
        "id": "make_complexity_limit",
        "keywords": ["make.com", "integromat"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "complex logic", "branching logic", "nested loops",
                "advanced workflow"
            ]),
            "flag":    "TECHNICAL_LIMITATION",
            "message": "Make.com has operation limits per plan (10k-800k/month). Complex scenarios consume operations fast. Budget accordingly.",
        }],
    },

    {
        "id": "openai_api_rate_limits",
        "keywords": ["openai api", "gpt-4", "chatgpt api", "gpt-3.5"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "unlimited", "no limit", "thousands per minute",
                "instant response", "very fast", "no delay"
            ]),
            "flag":    "TECHNICAL_LIMITATION",
            "message": "OpenAI API has rate limits (RPM/TPM) and latency (2-30s for GPT-4). High volume needs tier upgrades + queueing.",
        }],
    },

    {
        "id": "wix_seo_limit",
        "keywords": ["wix"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "seo", "search ranking", "google ranking",
                "organic traffic", "technical seo"
            ]),
            "flag":    "TECHNICAL_LIMITATION",
            "message": "Wix has historical SEO limitations (JS rendering, URL structure, page speed). For SEO-critical projects, use WordPress or static site.",
        }],
    },

    {
        "id": "wix_vendor_lockin",
        "keywords": ["wix"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "migrate", "migration", "export data",
                "move away", "switch platform", "leave wix"
            ]),
            "flag":    "TECHNICAL_LIMITATION",
            "message": "Wix has vendor lock-in: CMS content, blogs, member data are difficult to export. Warn clients before committing.",
        }],
    },

    {
        "id": "shopify_custom_checkout",
        "keywords": ["shopify"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "custom checkout", "modify checkout",
                "checkout page design", "checkout ui"
            ]),
            "flag":    "TECHNICAL_LIMITATION",
            "message": "Shopify restricts checkout customization to Shopify Plus ($2000+/month). Standard plans cannot fully modify checkout.",
        }],
    },

    {
        "id": "shopify_recurring_costs",
        "keywords": ["shopify"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "one time", "one-time", "no monthly",
                "no subscription", "no recurring"
            ]),
            "flag":    "TECHNICAL_LIMITATION",
            "message": "Shopify has mandatory recurring costs: $39-$399+/month + transaction fees + app subscriptions. No one-time model.",
        }],
    },

    {
        "id": "wordpress_shared_hosting",
        "keywords": ["wordpress", "wp"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "shared hosting", "bluehost", "hostgator",
                "namecheap hosting", "godaddy hosting"
            ]),
            "flag":    "TECHNICAL_LIMITATION",
            "message": "Shared hosting is not suitable for production WordPress. Recommend managed hosting (Kinsta, WP Engine, Cloudways).",
        }],
    },

    {
        "id": "squarespace_custom_functionality",
        "keywords": ["squarespace"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "custom plugin", "custom app", "database",
                "user login", "membership", "custom api", "backend"
            ]),
            "flag":    "TECHNICAL_LIMITATION",
            "message": "Squarespace has limited custom functionality. No plugins, no backend. For complex features use WordPress or Webflow.",
        }],
    },

    {
        "id": "webflow_cms_limit",
        "keywords": ["webflow"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "10000", "10,000", "thousands of products",
                "large catalog", "massive inventory"
            ]),
            "flag":    "TECHNICAL_LIMITATION",
            "message": "Webflow CMS is capped at 10,000 items per collection and 20 collections per site. Use Shopify for large catalogs.",
        }],
    },

    {
        "id": "flutter_native_features",
        "keywords": ["flutter"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "bluetooth", "nfc", "bgm processing", "deep ar",
                "advanced camera", "ble"
            ]),
            "flag":    "TECHNICAL_LIMITATION",
            "message": "Flutter has limited support for some native features (advanced Bluetooth, NFC variants). May need native bridges.",
        }],
    },

    {
        "id": "instagram_api_restrictions",
        "keywords": ["instagram api", "instagram integration"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "scrape", "auto post", "auto follow",
                "auto dm", "mass dm", "bot"
            ]),
            "flag":    "TECHNICAL_LIMITATION",
            "message": "Instagram API is restricted. No auto-DM, no auto-follow. Business API requires app review. Most scraping violates ToS.",
        }],
    },

    {
        "id": "whatsapp_business_approval",
        "keywords": ["whatsapp", "whatsapp business", "twilio whatsapp"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "instant approval", "quick approval", "marketing message",
                "broadcast", "bulk message", "promotional"
            ]),
            "flag":    "TECHNICAL_LIMITATION",
            "message": "WhatsApp Business API needs Meta template approval (3-7 days). Marketing templates are heavily restricted.",
        }],
    },

    # ═══════════════════════════════════════════════════════════
    # MISSING INFO RULES
    # ═══════════════════════════════════════════════════════════

    {
        "id": "missing_budget",
        "keywords": ["price", "quote", "quotation", "estimate", "cost",
                     "how much"],
        "checks": [{
            "trigger": lambda txt: (
                _contains_any(txt, ["how much", "price", "cost", "estimate", "quote"])
                and not _has_budget_mentioned(txt)
            ),
            "flag":    "MISSING_BUDGET",
            "message": "Client asks for pricing but did not specify their budget. Ask for budget range before quoting.",
        }],
    },

    {
        "id": "missing_timeline",
        "keywords": ["develop", "build", "create", "need", "want", "project"],
        "checks": [{
            "trigger": lambda txt: (
                len(txt) > 100
                and not _has_timeline_mentioned(txt)
                and _contains_any(txt, ["project", "develop", "build", "need"])
            ),
            "flag":    "MISSING_TIMELINE",
            "message": "Client did not specify deadline or timeline. Clarify expected delivery date before scoping.",
        }],
    },

    # ═══════════════════════════════════════════════════════════
    # LEGAL / IP / ETHICAL RULES
    # ═══════════════════════════════════════════════════════════

    {
        "id": "brand_concealment",
        "keywords": ["never name", "never mention", "never repeat",
                     "without naming", "hide the brand", "secretly",
                     "must not mention", "must never mention",
                     "don't mention", "do not mention"],
        "checks": [{
            "trigger": lambda txt: True,
            "flag":    "LEGAL_RISK",
            "message": "Client wants AI to hide/conceal brand names. Legal/IP implications. Cannot guarantee 100% LLM compliance due to hallucination risk.",
        }],
    },

    {
        "id": "competitor_scraping",
        "keywords": ["scrape", "scraping", "extract from competitor",
                     "copy from", "duplicate from", "clone website"],
        "checks": [{
            "trigger": lambda txt: True,
            "flag":    "LEGAL_RISK",
            "message": "Client asks for scraping/cloning competitor content. Verify ToS, copyright, and legal compliance before accepting.",
        }],
    },

    {
        "id": "fake_reviews",
        "keywords": ["fake review", "auto review", "generate review",
                     "boost rating", "auto rating", "fake testimonial"],
        "checks": [{
            "trigger": lambda txt: True,
            "flag":    "ETHICAL_RISK",
            "message": "Client asks for fake reviews/ratings. Violates most platform ToS and consumer protection laws.",
        }],
    },

    {
        "id": "spam_automation",
        "keywords": ["mass email", "bulk email", "cold email blast",
                     "spam", "mass dm", "bulk dm"],
        "checks": [{
            "trigger": lambda txt: True,
            "flag":    "ETHICAL_RISK",
            "message": "Mass/bulk messaging risks ToS violations, ESP suspension, GDPR/CAN-SPAM penalties. Recommend opt-in marketing instead.",
        }],
    },

    {
        "id": "gdpr_sensitive_data",
        "keywords": ["eu", "europe", "germany", "france", "gdpr",
                     "european users"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "user data", "customer data", "personal data",
                "store data", "collect data"
            ]),
            "flag":    "LEGAL_RISK",
            "message": "EU/GDPR compliance required. Need data processing agreement, EU hosting, consent management, and right-to-delete features.",
        }],
    },

    # ═══════════════════════════════════════════════════════════
    # COMMUNICATION / WORKFLOW RISKS
    # ═══════════════════════════════════════════════════════════

    {
        "id": "language_barrier",
        "keywords": ["english is poor", "poor english", "limited english",
                     "bad english", "english not good", "my english",
                     "use translation", "translation tool"],
        "checks": [{
            "trigger": lambda txt: True,
            "flag":    "COMMUNICATION_RISK",
            "message": "Client mentioned limited English. Risk of scope misunderstanding. Document everything in writing and confirm requirements explicitly.",
        }],
    },

    {
        "id": "scope_creep_simple",
        "keywords": ["simple", "easy", "quick", "just a small", "basic",
                     "minor", "straightforward", "shouldn't take long"],
        "checks": [{
            "trigger": lambda txt: (
                _contains_any(txt, ["simple", "easy", "quick", "straightforward"])
                and len(txt) > 300
            ),
            "flag":    "SCOPE_RISK",
            "message": "Client describes project as 'simple' but requirements suggest otherwise. Watch for scope creep and unrealistic expectations.",
        }],
    },

    {
        "id": "no_zoom_meeting",
        "keywords": ["no call", "no meeting", "no zoom", "chat only",
                     "text only", "only chat"],
        "checks": [{
            "trigger": lambda txt: True,
            "flag":    "COMMUNICATION_RISK",
            "message": "Client refuses calls/meetings. Risk of miscommunication for complex projects. Insist on at least one kick-off call.",
        }],
    },

    # ═══════════════════════════════════════════════════════════
    # HALLUCINATION CHECKS
    # ═══════════════════════════════════════════════════════════

    {
        "id": "non_existent_apis",
        "keywords": ["api"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "autocalls api", "fiverr api", "instagram api direct",
                "tiktok api direct", "facebook api unrestricted",
                "linkedin scraping api"
            ]),
            "flag":    "HALLUCINATED_API",
            "message": "This API may not exist or has heavy restrictions. Verify availability and access requirements before promising integration.",
        }],
    },

    {
        "id": "unrealistic_ai_capability",
        "keywords": ["ai", "gpt", "llm"],
        "checks": [{
            "trigger": lambda txt: _contains_any(txt, [
                "100% accurate", "no hallucination", "100% accuracy",
                "perfect ai", "always correct", "never wrong"
            ]),
            "flag":    "UNREALISTIC_EXPECTATION",
            "message": "Client expects 100% AI accuracy. No LLM provides this. Set expectations: 85-95% accuracy with human review for critical cases.",
        }],
    },
]


# ── Core Functions ───────────────────────────────────────────

def check(text: str) -> list[dict]:
    """
    Apply all rules to text. Returns list of violations found.
    Each rule fires only once (deduplicates by rule_id).
    """
    text_lower = text.lower()
    violations = []
    seen_rules = set()

    for rule in RULES:
        if rule["id"] in seen_rules:
            continue

        # Skip rule if no keywords match
        if not any(kw in text_lower for kw in rule["keywords"]):
            continue

        # Run checks
        for chk in rule["checks"]:
            try:
                if chk["trigger"](text):
                    violations.append({
                        "rule_id": rule["id"],
                        "flag":    chk["flag"],
                        "message": chk["message"],
                    })
                    seen_rules.add(rule["id"])
                    break
            except Exception:
                continue

    return violations


def format_warnings(violations: list[dict]) -> str:
    """Format violations as readable text."""
    if not violations:
        return ""
    lines = ["\n\n--- NSR Warnings ---"]
    for v in violations:
        lines.append(f"[{v['flag']}] {v['message']}")
    return "\n".join(lines)


def is_safe(text: str) -> bool:
    """Returns True if no violations."""
    return len(check(text)) == 0


def get_rules_by_category() -> dict:
    """Return rules grouped by their flag category (for debugging/admin)."""
    grouped = {}
    for rule in RULES:
        for chk in rule["checks"]:
            flag = chk["flag"]
            if flag not in grouped:
                grouped[flag] = []
            grouped[flag].append(rule["id"])
    return grouped