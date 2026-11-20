import asyncio
import os
import sys
from pathlib import Path
from groq import AsyncGroq

backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))
os.environ["PYTHONPATH"] = str(backend_dir)

from app.core.config import get_settings
from app.services.llm_service import FRIEND_SYSTEM_PROMPT, _parse_llm_json

async def check_gpt_oss_quality():
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

    resp = await client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        max_tokens=1000,
        temperature=0.3,
        reasoning_effort="low"
    )
    raw = resp.choices[0].message.content
    parsed = _parse_llm_json(raw, test_user_text)
    print("Parsed output from openai/gpt-oss-20b:")
    print("Alex response:", parsed.get("response_text"))
    print("Corrections count:", len(parsed.get("corrections", [])))
    for c in parsed.get("corrections", []):
        print(f"  {c.get('original')} -> {c.get('corrected')} | {c.get('spoken_comparison')} | {c.get('explanation')}")

if __name__ == "__main__":
    asyncio.run(check_gpt_oss_quality())
