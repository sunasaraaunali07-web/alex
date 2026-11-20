import asyncio
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("backend"))

from groq import AsyncGroq
from app.core.config import get_settings
from app.services.llm_service import _parse_llm_json

PROMPT_TEST = """You are Alex, a warm, intelligent English-speaking friend helping the user practise conversational English.

## RESPONSE FORMAT
Respond with ONLY a valid JSON object:
{
  "response_text": "your complete, natural and intelligent conversational reply answering the user's question thoroughly",
  "corrections": [
    {
      "original": "exact erroneous phrase from the LATEST user message",
      "corrected": "minimally corrected phrase",
      "spoken_comparison": "explicit comparison e.g. In that sentence, say 'X' instead of 'Y'",
      "emphasized_word": "WORD",
      "full_original_sentence": "original sentence",
      "full_corrected_sentence": "corrected sentence",
      "explanation": "brief reason",
      "type": "grammar",
      "complexity": 1
    }
  ]
}
"""

async def main():
    s = get_settings()
    c = AsyncGroq(api_key=s.GROQ_API_KEY)
    t = "Yesterday I go to college and my professor tell me about data analytics. What kind of job I can get with data analytics?"
    uc = f"{t}\n\n[Evaluate the latest message for grammar corrections AND provide your complete conversational response answering the user's questions in the required JSON object format]"
    resp = await c.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": PROMPT_TEST},
            {"role": "user", "content": uc}
        ],
        max_tokens=1000,
        temperature=0.3
    )
    raw = resp.choices[0].message.content
    print("RAW CONTENT:\n", raw)
    parsed = _parse_llm_json(raw, t)
    print("\nPARSED KEYS:", list(parsed.keys()))
    print("RESPONSE TEXT:", parsed.get("response_text"))
    print("CORRECTIONS COUNT:", len(parsed.get("corrections", [])))

if __name__ == "__main__":
    asyncio.run(main())
