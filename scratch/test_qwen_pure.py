import asyncio
import time
import os
import sys
from pathlib import Path
from groq import AsyncGroq

backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))
os.environ["PYTHONPATH"] = str(backend_dir)

from app.core.config import get_settings
from app.services.llm_service import FRIEND_SYSTEM_PROMPT

async def test_qwen_pure():
    settings = get_settings()
    client = AsyncGroq(api_key=settings.GROQ_API_KEY.strip())
    
    test_user_text = "Yesterday I go to college and because I have to submit my assignment."
    
    prompt = (
        f"{test_user_text}\n\n"
        "[MANDATORY AUDIT RULES:\n"
        "1. SHORT PHRASES ONLY (2 TO 5 WORDS MAXIMUM): In 'original' and 'corrected', output ONLY the error phrase with 1-2 words of context (e.g. 'what I trying' -> 'what I am trying', 'still not able' -> 'am still not able', 'buy apple' -> 'bought an apple', 'reach at home' -> 'reached home'). ABSOLUTELY NEVER dump the whole sentence into 'original' or 'corrected'!\n"
        "2. Split multi-error sentences: Each distinct mistake gets its OWN separate short card. NEVER combine multiple mistakes into one long sentence.\n"
        "3. Spoken comparison: Strictly 'In that sentence, say X instead of Y.' (Do NOT include 'because' or grammar explanations in spoken_comparison).\n"
        "4. Meta-language awareness: When user discusses language concepts, gives feedback, or cites example phrases (e.g. 'buy bought', 'using she going'), recognize they are discussing examples. Do NOT treat quoted examples as mistakes.\n"
        "5. Alex is a warm conversational friend: Reply purely conversationally to the user in 'response_text'. NEVER give coding, HTML, or CSS instructions! NEVER lecture on grammar in 'response_text'!\n"
        "Output valid JSON only.]"
    )

    messages = [
        {"role": "system", "content": FRIEND_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    for i in range(3):
        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=messages,
            max_tokens=600,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        dur = int((time.perf_counter() - t0) * 1000)
        content = resp.choices[0].message.content
        print(f"Run {i+1}: {dur}ms | Tokens: {resp.usage.completion_tokens} (prompt: {resp.usage.prompt_tokens})")

if __name__ == "__main__":
    asyncio.run(test_qwen_pure())
