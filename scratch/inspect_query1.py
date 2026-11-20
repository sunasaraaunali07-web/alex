import asyncio
import json
from groq import AsyncGroq
from app.core.config import get_settings
from app.services.llm_service import FRIEND_SYSTEM_PROMPT

async def inspect_query1():
    settings = get_settings()
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    q = "I am confused about my career. Can you compare Data Analytics, Machine Learning and AI Application Development? Which one should I start with, and can you give me a 6 month roadmap with projects and resources, and how to evaluate internship readiness?"
    user_content = f"{q}\n\n[Analyze ONLY this latest message for any corrections. Respond strictly in the required JSON object format]"
    messages = [
        {"role": "system", "content": FRIEND_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    res = await client.chat.completions.create(
        model=settings.GROQ_LLM_MODEL,
        messages=messages,
        max_tokens=800,
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    raw = res.choices[0].message.content
    print("COMPLETE RAW RESPONSE:")
    print(raw)
    data = json.loads(raw)
    print("\nKeys:", data.keys())
    print("response_text:", repr(data.get("response_text")))

if __name__ == "__main__":
    asyncio.run(inspect_query1())
