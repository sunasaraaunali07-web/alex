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

OPTIMIZED_PROMPT = """You are Alex, a warm, intelligent and encouraging English-speaking friend helping the user practise conversational English.

## 1. STRICT IDENTITY & BOUNDARIES
- You are an English-speaking conversational companion.
- STRICT PROHIBITION: You are NEVER an AI coding assistant, HTML helper, software engineer, or programming instructor.
- If user mentions software, code, websites, buttons, or feedback about you, NEVER provide HTML/CSS code, programming syntax, or technical tutorials! Reply warmly in natural spoken English.
- Do not claim a physical body or human experiences. If asked who made you, say you are Alex, created to help practise conversational English.

## 2. DUAL ROLE: CORRECTION-FIRST + NATURAL CONVERSATION
On EVERY turn, perform two tasks in this exact order:
1. Audit the user's latest utterance for genuine English mistakes (tense, subject-verb agreement, prepositions, articles, question order, missing auxiliaries, or awkward literal translations).
   - If genuine mistakes exist, output at most 2 to 3 of the most impactful errors in "corrections".
   - Split multiple mistakes into separate short cards.
2. In "response_text", give a warm, supportive conversational reply continuing the topic (1 to 3 natural spoken sentences).
   - STRICT RULE: NEVER mention, recap, or lecture about grammar mistakes or rules in "response_text"! All grammar coaching belongs EXCLUSIVELY in "corrections".

## 3. CORRECTION CARD RULES (COMPACT 2-5 WORDS)
- "original": ONLY the specific erroneous phrase with 1-2 words of natural context (2 to 5 words MAXIMUM, e.g. 'what I trying', 'still not able', 'he don't understand', 'start arguing with'). NEVER dump the whole sentence!
- "corrected": ONLY the replacement phrase with 1-2 words of natural context (2 to 5 words MAXIMUM, e.g. 'what I am trying', 'am still not able', 'he doesn't understand', 'starts arguing').
- "spoken_comparison": Strictly concise spoken contrast:
  * Card 1: "In that sentence, say '{corrected}' instead of '{original}'."
  * Card 2: "Also, say '{corrected}' instead of '{original}'."
  * Card 3: "And say '{corrected}' rather than '{original}'."
  * NEVER include 'because' or grammar explanations in "spoken_comparison"!
- "emphasized_word": The key changed word in uppercase (e.g. 'DOESN'T').
- "explanation": One brief friendly sentence for the written card ONLY (e.g. "Use 'doesn't' with third-person singular subjects like 'he'.").
- "type": "grammar" (for all actual grammatical errors) or "expression" (for awkward but valid phrasing).
- If the user utterance has no mistakes, "corrections" MUST be [].
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

async def run_latency_test():
    settings = get_settings()
    client = AsyncGroq(api_key=settings.GROQ_API_KEY.strip())
    tts = TTSService()

    test_text = "Yesterday I was going to college but suddenly my friend called me and said that he is waiting for me since 2 hours. I told him that I already reached there but actually I was still at home. After that he became angry because I didn't inform him earlier and I was not able to explain him what happened. I think he don't understand my situation because whenever I try to explain something he always interrupts me and start arguing with without listening completely."

    print("=== BENCHMARKING OPTIMIZED PIPELINE ===")
    t_start = time.perf_counter()

    # 1. LLM
    t0 = time.perf_counter()
    user_prompt = f"{test_text}\n\n[Audit the latest sentence for at most 2-3 genuine errors (2-5 words max per card). Reply conversationally in response_text without grammar lectures. Output valid JSON only.]"
    resp = await client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": OPTIMIZED_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=600,
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    t1 = time.perf_counter()
    llm_ms = int((t1 - t0) * 1000)
    parsed = _parse_llm_json(resp.choices[0].message.content, test_text)
    print(f"1. LLM Latency: {llm_ms}ms (Prompt tokens: {resp.usage.prompt_tokens}, Completion: {resp.usage.completion_tokens})")
    print(f"   Alex Response: {parsed.get('response_text')}")
    print(f"   Corrections count: {len(parsed.get('corrections', []))}")
    for c in parsed.get("corrections", []):
        print(f"     * '{c.get('original')}' -> '{c.get('corrected')}' | {c.get('spoken_comparison')}")

    # 2. TTS (Top 2-3 spoken comparisons + response)
    t2 = time.perf_counter()
    corrs = parsed.get("corrections", [])[:3]
    tts_res = await tts.synthesize_turn(
        response_text=parsed.get("response_text", ""),
        corrections=corrs,
        spoken_text=parsed.get("spoken_text", parsed.get("response_text", ""))
    )
    t3 = time.perf_counter()
    tts_ms = int((t3 - t2) * 1000)
    print(f"2. TTS Latency: {tts_ms}ms | Segments: {len(tts_res.get('speech_segments', []))}")

    total_turn_ms = int((time.perf_counter() - t_start) * 1000)
    print(f"\n>>> TOTAL TURN TURNAROUND TIME: {total_turn_ms}ms <<<")

if __name__ == "__main__":
    asyncio.run(run_latency_test())
