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

async def compare_llm_speeds():
    settings = get_settings()
    client = AsyncGroq(api_key=settings.GROQ_API_KEY.strip())
    
    test_user_text = "Yesterday I was going to college but suddenly my friend called me and said that he is waiting for me since 2 hours. I told him that I already reached there but actually I was still at home. After that he became angry because I didn't inform him earlier and I was not able to explain him what happened. I think he don't understand my situation because whenever I try to explain something he always interrupts me and start arguing with without listening completely."
    
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

    for model in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
        try:
            t0 = time.perf_counter()
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.3
            }
            if "openai" in model:
                kwargs["reasoning_effort"] = "low"
            else:
                kwargs["response_format"] = {"type": "json_object"}
            resp = await client.chat.completions.create(**kwargs)
            dur = int((time.perf_counter() - t0) * 1000)
            content = resp.choices[0].message.content
            print(f"Model: {model} -> {dur}ms | Output len: {len(content)}")
        except Exception as e:
            print(f"Model: {model} -> FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(compare_llm_speeds())
