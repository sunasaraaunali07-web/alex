import asyncio
import json
import time
from groq import AsyncGroq
from app.core.config import get_settings

async def test_deep_question_structured():
    settings = get_settings()
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    prompt = """
You are Alex, an intelligent, knowledgeable and friendly English-speaking companion.

## RESPONSE FORMAT
Respond with ONLY a valid JSON object:
{
  "corrections": [],
  "response_text": "your complete, well-structured response"
}

## CONVERSATIONAL INTELLIGENCE RULES
- Comprehend every part of the user's prompt and answer all requested points systematically.
- If the user asks a casual question, give a concise, engaging reply (2–3 sentences).
- If the user asks a multi-part, technical, or detailed question, provide a complete, well-structured response covering every requested point (e.g. comparisons, required skills, project ideas, practical steps, roadmaps).
- Keep your answers high-density, well-organized, and direct (approx. 250–400 words for multi-part questions) so every part is thoroughly covered without rambling.
- Never ignore sub-questions. Provide realistic recommendations with reasoning.
- Avoid generic motivational filler; jump straight into the knowledgeable, engaging substance.
"""

    user_query = (
        "Can you compare Data Analytics, Machine Learning, and AI Application Development? "
        "Please cover career opportunities, required skills, projects to build, and a realistic 6-month roadmap."
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_query}
    ]

    print("Sending deep question with high-density structuring...")
    start = time.perf_counter()
    res = await client.chat.completions.create(
        model=settings.GROQ_LLM_MODEL,
        messages=messages,
        max_tokens=850,
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    duration = time.perf_counter() - start
    content = res.choices[0].message.content
    print(f"SUCCESS in {duration:.2f}s!")
    data = json.loads(content)
    resp = data.get("response_text", "")
    print(f"\nResponse Length: {len(resp)} chars, {len(resp.split())} words")
    print("\n--- RESPONSE TEXT ---")
    print(resp)
    print("\nTokens used:", res.usage.completion_tokens, "out of", res.usage.total_tokens)

if __name__ == "__main__":
    asyncio.run(test_deep_question_structured())
