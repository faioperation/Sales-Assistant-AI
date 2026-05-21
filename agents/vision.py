"""
Vision Agent
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import OPENAI_API_KEY, LLM_MODEL

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.4,
)

SYSTEM_PROMPT = """
You are a visual analyst. Analyze images and provide:
- Description of what's in the image
- Any text visible (OCR)
- Sales/business relevance
- Actionable insights

Be precise. Do NOT make up content.
"""


def analyze(image_data: str, user_question: str = "") -> dict:
    if not image_data:
        return {
            "sections": [{"id": "vision_error", "title": "Error",
                          "content": "No image provided"}],
            "nsr_warnings": [],
            "raw": "",
        }

    if image_data.startswith("http") or image_data.startswith("data:image"):
        image_url = image_data
    else:
        image_url = f"data:image/png;base64,{image_data}"

    question = user_question or "Analyze this image thoroughly."

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=[
            {"type": "text",      "text": question},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]),
    ]

    try:
        response = llm.invoke(messages)
        output   = response.content
        sections = [
            {"id": "image_analysis", "title": "Image Analysis", "content": output},
        ]
        return {
            "sections":     sections,
            "nsr_warnings": [],
            "raw":          output,
        }
    except Exception as e:
        return {
            "sections":     [{"id": "vision_error", "title": "Error",
                              "content": f"Error: {str(e)}"}],
            "nsr_warnings": [],
            "raw":          "",
            "error":        str(e),
        }