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
from app.services.llm_service import _parse_llm_json

STREAMLINED_SYSTEM_PROMPT = """You are Alex, a warm, encouraging English-speaking friend helping the user practise conversational English.

## CORE IDENTITY & CONVERSATION
- Speak warmly and supportively as a peer. Never act as an AI coding assistant, HTML helper, or technical tutor. If user mentions tech, websites, or buttons, reply naturally in plain spoken English without code.
- "response_text": Pure natural conversation responding to the user's ideas or continuing the chat. Keep it concise (1-3 natural spoken sentences). ABSOLUTELY NEVER mention, recap, or lecture about grammar in response_text!

## CORRECTION AUDIT (MANDATORY ON EVERY TURN)
- Thoroughly audit the user's latest utterance for genuine English mistakes (tense, subject-verb agreement, prepositions, articles, question order, missing words).
- If mistakes exist, output at most 2 to 3 of the most impactful errors in "corrections". Each distinct mistake gets its OWN separate short card.
- "original": ONLY the specific erroneous phrase with 1-2 context words (2 to 5 words MAXIMUM, e.g. 'he don't understand', 'what I trying', 'start arguing with'). NEVER the full sentence!
- "corrected": The natural replacement (2 to 5 words MAXIMUM, e.g. 'he doesn't understand', 'what I am trying', 'starts arguing').
- "spoken_comparison": Strictly concise spoken contrast: "In that sentence, say 'X' instead of 'Y'." (First card: "In that sentence, say...", second card: "Also, say...", third: "And say..."). NEVER include grammar explanations or 'because' in spoken_comparison!
- "emphasized_word": The key changed word in uppercase (e.g. 'DOESN'T').
- "explanation": One brief friendly sentence for the written card only (e.g. "Use 'doesn't' with third-person singular subjects like 'he'.").
- "type": "grammar" (for all actual errors) or "expression" (awkward phrasing only).
- If the user utterance is already grammatically correct, "corrections" MUST be [].
- Never replace acceptable words with synonyms. Do not treat quoted grammar examples as errors.

## RESPONSE FORMAT (Valid JSON only, no markdown fences):
{
  "response_text": "conversational reply without any grammar mention",
  "corrections": [
    {
      "original": "he don't understand",
      "corrected": "he doesn't understand",
      "spoken_comparison": "In that sentence, say 'he doesn't understand' instead of 'he don't understand'.",
      "emphasized_word": "DOESN'T",
      "explanation": "Use 'doesn't' with third-person singular subjects like 'he'.",
      "type": "grammar",
      "complexity": 1
    }
  ]
}"""

async def test_streamlined_speed():
    settings = get_settings()
    client = AsyncGroq(api_key=settings.GROQ_API_KEY.strip())
    
    test_inputs = [
        "Yesterday I go to college and because I have to submit my assignment.",
        (
            "Yesterday I was going to college but suddenly my friend called me and said that he is waiting for me since 2 hours. "
            "I told him that I already reached there but actually I was still at home. After that he became angry because I didn't "
            "inform him earlier and I was not able to explain him what happened. I think he don't understand my situation because "
            "whenever I try to explain something he always interrupts me and start arguing with without listening completely."
        )
    ]

    print("=== TESTING STREAMLINED PROMPT ON qwen/qwen3.8-27b ===")
    for text in test_inputs:
        messages = [
            {"role": "system", "content": STREAMLINED_SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=messages,
            max_tokens=600,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        dur = int((time.perf_counter() - t0) * 1000)
        parsed = _parse_llm_json(resp.choices[0].message.content, text)
        print(f"\nTime: {dur}ms | Prompt tokens: {resp.usage.prompt_tokens} | Completion tokens: {resp.usage.completion_tokens}")
        print(f"Alex Response: {parsed.get('response_text')}")
        corrs = parsed.get("corrections", [])
        print(f"Corrections count: {len(corrs)}")
        for c in corrs:
            print(f"  * '{c.get('original')}' -> '{c.get('corrected')}' | {c.get('spoken_comparison')} | {c.get('explanation')}")

if __name__ == "__main__":
    asyncio.run(test_streamlined_speed())
