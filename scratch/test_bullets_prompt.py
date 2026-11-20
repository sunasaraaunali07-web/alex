import asyncio
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("backend"))

from groq import AsyncGroq
from app.core.config import get_settings
from app.services.llm_service import FRIEND_SYSTEM_PROMPT, _parse_llm_json

PROMPT_BULLETS = FRIEND_SYSTEM_PROMPT + """
- For structured plans, roadmaps, and comparisons, use concise numbered or bulleted lists instead of markdown tables.
"""

async def test():
    s = get_settings()
    c = AsyncGroq(api_key=s.GROQ_API_KEY)
    
    career_question = (
        "Can you compare Data Analytics, Machine Learning, and AI Application Development for me? "
        "Which starting path would you recommend for me and why? "
        "Please provide a realistic six-month roadmap, including specific projects to build, "
        "free learning resources, and explain how I can know if I am internship-ready and how to evaluate it."
    )
    
    user_content = f"{career_question}\n\n[Evaluate the latest message for grammar corrections AND provide your complete conversational response answering the user's questions in the required JSON object format]"
    
    resp = await c.chat.completions.create(
        model=s.GROQ_LLM_MODEL,
        messages=[
            {"role": "system", "content": PROMPT_BULLETS},
            {"role": "user", "content": user_content}
        ],
        max_tokens=1400,
        temperature=0.3
    )
    raw = resp.choices[0].message.content
    print("Finish reason:", resp.choices[0].finish_reason)
    parsed = _parse_llm_json(raw, career_question)
    response_text = parsed.get("response_text", "")
    print(f"\nResponse length: {len(response_text)} chars")
    print("\n--- Response Text ---\n", response_text)
    
    lower = response_text.lower()
    print("\n--- Checks ---")
    print("Compares careers:", "data analytics" in lower and "machine learning" in lower and "ai" in lower)
    print("Recommendation:", "recommend" in lower or "start" in lower)
    print("Roadmap:", "roadmap" in lower or "month" in lower)
    print("Projects/Resources:", "project" in lower and ("resource" in lower or "free" in lower or "course" in lower))
    print("Internship readiness:", "intern" in lower or "ready" in lower or "evaluate" in lower)

if __name__ == "__main__":
    asyncio.run(test())
