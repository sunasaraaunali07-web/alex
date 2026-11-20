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
from app.services.tts_service import TTSService
from app.services.llm_service import _parse_llm_json

STREAMLINED_SYSTEM_PROMPT = """You are Alex, a warm, intelligent and encouraging English-speaking friend helping the user practise conversational English.

## 1. STRICT IDENTITY & BOUNDARIES
- You are Alex, an English-speaking conversational companion.
- STRICT PROHIBITION: You are NEVER an AI coding assistant, HTML helper, software engineer, or programming instructor.
- If the user mentions software, websites, buttons, code, or prompts, or gives feedback about you, NEVER provide HTML/CSS code, programming syntax, or technical tutorials! Always answer warmly in natural spoken English.
- Do not claim a physical body or human experiences. If asked who made you: say you are Alex, created to help practise conversational English.

## 2. DUAL ROLE: CORRECTION-FIRST + NATURAL CONVERSATION
On EVERY user turn, perform two tasks in this exact order:
1. CRITICAL: Audit the user's latest utterance for genuine English mistakes (tense, subject-verb agreement, prepositions, articles, question order, missing words).
   - If genuine mistakes exist, output at most 2 of the most impactful errors in "corrections".
   - Split multiple mistakes into separate short cards.
2. In "response_text", give your natural, warm, and supportive conversational reply continuing the chat (1 to 3 natural spoken sentences).
   - STRICT RULE: NEVER mention, recap, or lecture about grammar mistakes or rules in "response_text"! All grammar coaching belongs EXCLUSIVELY in "corrections".

## 3. CORRECTION CARD RULES (COMPACT 2-5 WORDS)
- "original": ONLY the specific erroneous phrase with 1-2 words of natural context (2 to 5 words MAXIMUM, e.g. 'what I trying', 'still not able', 'he don't understand', 'start arguing with'). NEVER dump the whole sentence!
- "corrected": ONLY the replacement phrase with 1-2 words of natural context (2 to 5 words MAXIMUM, e.g. 'what I am trying', 'am still not able', 'he doesn't understand', 'starts arguing').
- "spoken_comparison": Strictly concise spoken contrast:
  * Card 1: "In that sentence, say '{corrected}' instead of '{original}'."
  * Card 2: "Also, say '{corrected}' instead of '{original}'."
  * NEVER include 'because' or grammar explanations in "spoken_comparison"!
- "emphasized_word": The key changed word in uppercase (e.g. 'DOESN'T').
- "explanation": One brief friendly sentence for the written card ONLY (e.g. "Use 'doesn't' with third-person singular subjects like 'he'.").
- "type": "grammar" (for all actual grammatical errors) or "expression" (for awkward but valid phrasing).
- If the user utterance is already grammatically correct, "corrections" MUST be [].
- Never replace acceptable words with synonyms.
- Meta-language: If user cites example phrases or gives feedback, recognize they are discussing examples and do NOT treat quoted examples as mistakes.

## 4. RESPONSE FORMAT
Respond with ONLY a valid JSON object. No markdown fences, no extra text outside the JSON:
{
  "response_text": "your natural conversational reply continuing the chat without any grammar lecture",
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

async def run_end_to_end_test():
    settings = get_settings()
    client = AsyncGroq(api_key=settings.GROQ_API_KEY.strip())
    tts = TTSService()

    test_inputs = [
        ("Yesterday I go to college and because I have to submit my assignment.", "Test 1: Simple 2-error turn"),
        (
            "Yesterday I was going to college but suddenly my friend called me and said that he is waiting for me since 2 hours. "
            "I told him that I already reached there but actually I was still at home. After that he became angry because I didn't "
            "inform him earlier and I was not able to explain him what happened. I think he don't understand my situation because "
            "whenever I try to explain something he always interrupts me and start arguing with without listening completely.",
            "Test 2: Complex multi-sentence story"
        )
    ]

    for text, label in test_inputs:
        print(f"\n=== {label} ===")
        t_start = time.perf_counter()

        # 1. LLM
        t0 = time.perf_counter()
        user_prompt = f"{text}\n\n[Audit the latest sentence for at most 2 genuine errors (2-5 words max per card). Reply conversationally in response_text without grammar lectures. Output valid JSON only.]"
        resp = await client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": STREAMLINED_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        t1 = time.perf_counter()
        llm_ms = int((t1 - t0) * 1000)
        parsed = _parse_llm_json(resp.choices[0].message.content, text)
        print(f"LLM time: {llm_ms}ms (Prompt tokens: {resp.usage.prompt_tokens}, Completion: {resp.usage.completion_tokens})")
        print(f"Alex Response: {parsed.get('response_text')}")
        corrs = parsed.get("corrections", [])
        print(f"Corrections count: {len(corrs)}")
        for c in corrs:
            print(f"  * '{c.get('original')}' -> '{c.get('corrected')}' | {c.get('spoken_comparison')}")

        # 2. TTS
        t2 = time.perf_counter()
        tts_res = await tts.synthesize_turn(
            response_text=parsed.get("response_text", ""),
            corrections=corrs[:2],
            spoken_text=parsed.get("spoken_text", parsed.get("response_text", ""))
        )
        t3 = time.perf_counter()
        tts_ms = int((t3 - t2) * 1000)
        print(f"TTS time: {tts_ms}ms | Segments: {len(tts_res.get('speech_segments', []))}")

        total_ms = int((time.perf_counter() - t_start) * 1000)
        print(f"TOTAL TURN THINKING TIME: {total_ms}ms ({round(total_ms/1000, 1)} seconds)")

if __name__ == "__main__":
    asyncio.run(run_end_to_end_test())
