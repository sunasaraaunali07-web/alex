import asyncio
from groq import AsyncGroq
from app.core.config import get_settings

async def test_groq_models():
    settings = get_settings()
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    
    models = ["qwen/qwen3.8-27b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    for m in models:
        try:
            res = await client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "Hello! Give a 1-sentence reply."}],
                max_tokens=100
            )
            print(f"Model {m}: SUCCESS ({res.choices[0].message.content.strip()})")
        except Exception as e:
            print(f"Model {m}: FAILED ({e})")

if __name__ == "__main__":
    asyncio.run(test_groq_models())
