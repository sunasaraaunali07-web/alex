import asyncio
import json
from groq import AsyncGroq
from app.core.config import get_settings
from app.services.llm_service import FRIEND_SYSTEM_PROMPT

async def test_career_prompts():
    settings = get_settings()
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    test_queries = [
        "I am confused about my career. Can you compare Data Analytics, Machine Learning and AI Application Development? Which one should I start with, and can you give me a 6 month roadmap with projects and resources, and how to evaluate internship readiness?",
        "Can you give me detailed career guidance comparing Data Analytics, Machine Learning, and AI Application Development? Recommend the best starting path with reasoning, give a realistic 6-month roadmap, portfolio projects, free learning resources, and explain how I can know if I am internship-ready and how to evaluate it.",
        "Compare Data Analytics vs Machine Learning vs AI Application Development. Which starting path do you recommend and why? Give a realistic six-month roadmap, projects, free learning resources, and how to evaluate internship readiness.",
        "Hey Alex, I need career advice. What is the difference between Data Analytics, ML and AI App Development? Which should I pick first? Can you give me a roadmap for 6 months with projects, free resources and tell me when I will be internship ready?"
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n==================================================")
        print(f"TEST QUERY #{i}:")
        print(q)
        print("==================================================")

        user_content = f"{q}\n\n[Analyze ONLY this latest message for any corrections. Respond strictly in the required JSON object format]"
        messages = [
            {"role": "system", "content": FRIEND_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        try:
            res = await client.chat.completions.create(
                model=settings.GROQ_LLM_MODEL,
                messages=messages,
                max_tokens=800,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            raw = res.choices[0].message.content
            print("RAW RESPONSE FROM GROQ:")
            print(raw[:500] + "..." if len(raw) > 500 else raw)

            try:
                parsed = json.loads(raw)
                print("\nKeys in parsed JSON:", list(parsed.keys()))
                resp_text = parsed.get("response_text")
                print("Type of response_text:", type(resp_text))
                if isinstance(resp_text, dict):
                    print("WARNING: response_text is a DICTIONARY, not a string!")
                    print("Keys in response_text dict:", list(resp_text.keys()))
                elif isinstance(resp_text, list):
                    print("WARNING: response_text is a LIST, not a string!")
                else:
                    print("Length of response_text string:", len(str(resp_text or "")))
                    if not resp_text:
                        print(">>> BUG REPRODUCED: response_text is empty or None! <<<")
            except Exception as pe:
                print("JSON parse error:", pe)

        except Exception as e:
            print("Groq API error:", e)
            if hasattr(e, "body") and isinstance(e.body, dict):
                print("Error body:", e.body)

        await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(test_career_prompts())
