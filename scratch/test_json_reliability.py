import asyncio
import json
from groq import AsyncGroq
from app.core.config import get_settings
from app.services.llm_service import FRIEND_SYSTEM_PROMPT

async def test_prompts_without_json_flag():
    settings = get_settings()
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    queries = [
        "Yesterday I go to market and meet my friend.",
        "How are you today?",
        "Can you compare Data Analytics, Machine Learning, and AI Application Development for me? Which starting path would you recommend for me and why? Please provide a realistic six-month roadmap, including specific projects to build, free learning resources, and explain how I can know if I am internship-ready and how to evaluate it.",
        "I was wondering if you could help me prepare for a technical interview on Python and SQL."
    ]

    for q in queries:
        print(f"\n--- Testing Query: '{q[:40]}...' ---")
        user_content = f"{q}\n\n[Evaluate the latest message for grammar corrections AND provide your complete conversational response answering the user's questions in the required JSON object format]"
        res = await client.chat.completions.create(
            model=settings.GROQ_LLM_MODEL,
            messages=[
                {"role": "system", "content": FRIEND_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            max_tokens=900,
            temperature=0.3
        )
        raw = res.choices[0].message.content.strip()
        print("Raw output starts with:", repr(raw[:30]))
        print("Raw output ends with:", repr(raw[-30:]))
        # Check if valid JSON or extractable
        try:
            d = json.loads(raw)
            print("Direct JSON parse: SUCCESS, keys =", list(d.keys()))
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if 0 <= start < end:
                try:
                    d = json.loads(raw[start:end])
                    print("Substring JSON parse: SUCCESS, keys =", list(d.keys()))
                except Exception as e2:
                    print("Substring JSON parse failed:", e2)
            else:
                print("No closing brace, but content length =", len(raw))
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(test_prompts_without_json_flag())
