import asyncio
import json
import re
from groq import AsyncGroq
from app.core.config import get_settings
from app.services.llm_service import FRIEND_SYSTEM_PROMPT, _parse_llm_json

async def test_groq_handling():
    settings = get_settings()
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    
    career_question = (
        "Can you compare Data Analytics, Machine Learning, and AI Application Development for me? "
        "Which starting path would you recommend for me and why? "
        "Please provide a realistic six-month roadmap, including specific projects to build, "
        "free learning resources, and explain how I can know if I am internship-ready and how to evaluate it."
    )
    
    user_content = f"{career_question}\n\n[Evaluate the latest message for grammar corrections AND provide your complete conversational response answering the user's questions in the required JSON object format]"
    messages = [
        {"role": "system", "content": FRIEND_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    print("Testing call WITHOUT response_format...")
    resp = await client.chat.completions.create(
        model=settings.GROQ_LLM_MODEL,
        messages=messages,
        max_tokens=950,
        temperature=0.3
    )
    raw = resp.choices[0].message.content.strip()
    parsed = _parse_llm_json(raw, career_question)
    
    print("Parsed keys:", list(parsed.keys()))
    print("Response text length:", len(parsed.get("response_text", "")))
    print("\nFirst 400 chars:\n", parsed.get("response_text", "")[:400])
    print("\nLast 400 chars:\n", parsed.get("response_text", "")[-400:])

if __name__ == "__main__":
    asyncio.run(test_groq_handling())
