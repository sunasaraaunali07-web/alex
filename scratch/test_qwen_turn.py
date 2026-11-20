import asyncio
import json
from groq import AsyncGroq
from app.core.config import get_settings
from app.services.llm_service import FRIEND_SYSTEM_PROMPT

async def test_qwen_career_turn():
    settings = get_settings()
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    # Let's test with dialogue history as happens in a real conversation!
    # Turn 1: user introduces themselves
    # Turn 2: user asks complex career question
    history = [
        {"role": "user", "content": "Hi Alex, I am a 3rd year engineering student."},
        {"role": "assistant", "content": "Hey there! It's great to meet you. What kind of engineering are you studying?"}
    ]

    career_q = (
        "I'm studying computer engineering and I'm really confused about my future path. "
        "Can you compare Data Analytics, Machine Learning, and AI Application Development for me? "
        "Which starting path would you recommend for me and why? "
        "Please provide a realistic six-month roadmap, including specific projects to build, "
        "free learning resources, and explain how I can know if I am internship-ready and how to evaluate it."
    )

    # Exact line 537 from llm_service.py
    user_content = f"{career_q}\n\n[Analyze ONLY this latest message for any corrections. Respond strictly in the required JSON object format]"

    messages = [{"role": "system", "content": FRIEND_SYSTEM_PROMPT}]
    for t in history:
        messages.append({"role": t["role"], "content": t["content"]})
    messages.append({"role": "user", "content": user_content})

    print("Sending multi-turn career query with current prompt & line 537...")
    res = await client.chat.completions.create(
        model=settings.GROQ_LLM_MODEL,
        messages=messages,
        max_tokens=850,
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    raw = res.choices[0].message.content
    print("\nRAW OUTPUT FROM MODEL:")
    print(raw)
    data = json.loads(raw)
    print("\nKeys:", list(data.keys()))
    print("response_text value:", repr(data.get("response_text")))

if __name__ == "__main__":
    asyncio.run(test_qwen_career_turn())
