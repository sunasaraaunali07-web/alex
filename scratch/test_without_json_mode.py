import asyncio
import json
from groq import AsyncGroq
from app.core.config import get_settings
from app.services.llm_service import FRIEND_SYSTEM_PROMPT

async def test_without_groq_json_mode():
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

    print("Testing WITHOUT response_format={'type': 'json_object'}...")
    # Without the strict json_object constraint, Groq does NOT abort the request on token truncation!
    res = await client.chat.completions.create(
        model=settings.GROQ_LLM_MODEL,
        messages=messages,
        max_tokens=800,
        temperature=0.3
    )
    raw = res.choices[0].message.content
    print("SUCCESS! Received response (status 200) without 400 error!")
    print("Length of raw content:", len(raw))
    print("First 300 chars:\n", raw[:300])
    print("\nLast 300 chars:\n", raw[-300:])
    print("\nFinish reason:", res.choices[0].finish_reason)

if __name__ == "__main__":
    asyncio.run(test_without_groq_json_mode())
