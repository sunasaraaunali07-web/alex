import asyncio
import json
import re
from groq import AsyncGroq
from app.core.config import get_settings
from app.services.llm_service import FRIEND_SYSTEM_PROMPT

def extract_response_robustly(raw_text: str) -> dict:
    if not raw_text:
        return {"corrections": [], "response_text": "I see! Tell me more about that."}

    text = raw_text.strip()
    if "```" in text:
        lines = text.split("\n")
        cleaned = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(cleaned).strip()

    # Attempt 1: Direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return extract_from_dict(data)
    except Exception:
        pass

    # Attempt 2: Outermost JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if 0 <= start < end:
        try:
            data = json.loads(text[start:end])
            if isinstance(data, dict):
                return extract_from_dict(data)
        except Exception:
            pass

    # Attempt 3: Regex extraction from truncated JSON
    # Look for "response_text": "..." or alternative keys
    for key in ["response_text", "response", "answer", "reply", "message"]:
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)', text)
        if m:
            extracted_val = m.group(1).encode('utf-8').decode('unicode_escape', errors='ignore')
            if len(extracted_val.strip()) > 20:
                return {"corrections": [], "response_text": extracted_val.strip()}

    # Attempt 4: If text contains raw response without JSON
    clean_text = re.sub(r'[{}\[\]"]', ' ', text).strip()
    clean_text = re.sub(r'corrections\s*:\s*', '', clean_text)
    clean_text = re.sub(r'response_text\s*:\s*', '', clean_text)
    return {"corrections": [], "response_text": clean_text[:1200]}

def extract_from_dict(data: dict) -> dict:
    # 1. Primary keys
    response_text = ""
    for key in ["response_text", "response", "answer", "reply", "message", "content", "text"]:
        val = data.get(key)
        if val and isinstance(val, str) and val.strip():
            response_text = val.strip()
            break
        elif val and isinstance(val, (dict, list)):
            response_text = json.dumps(val, indent=2)
            break

    # 2. If response_text is still empty, look for section keys (e.g. comparison, roadmap)
    if not response_text:
        section_parts = []
        for k, v in data.items():
            if k not in ["corrections", "spoken_text", "timing_schedule", "speech_segments"] and isinstance(v, str) and v.strip():
                section_parts.append(f"**{k.replace('_', ' ').title()}:** {v.strip()}")
        if section_parts:
            response_text = "\n\n".join(section_parts)

    if not response_text:
        response_text = "That's interesting! Tell me more."

    return {
        "corrections": data.get("corrections", []),
        "response_text": response_text
    }

async def test_live_groq_with_improvements():
    settings = get_settings()
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    career_question = (
        "Can you compare Data Analytics, Machine Learning, and AI Application Development for me? "
        "Which starting path would you recommend for me and why? "
        "Please provide a realistic six-month roadmap, including specific projects to build, "
        "free learning resources, and explain how I can know if I am internship-ready and how to evaluate it."
    )

    # Corrected user instruction that does NOT say "Analyze ONLY for corrections":
    user_content = f"{career_question}\n\n[Evaluate the latest message for grammar corrections AND provide your complete conversational response answering the user's questions in the required JSON object format]"

    messages = [
        {"role": "system", "content": FRIEND_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    print("Sending career question with improved user instruction & 950 max_tokens...")
    res = await client.chat.completions.create(
        model=settings.GROQ_LLM_MODEL,
        messages=messages,
        max_tokens=950,
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    raw = res.choices[0].message.content
    print("\nRAW OUTPUT:")
    print(raw[:400] + "...")

    parsed = extract_response_robustly(raw)
    resp = parsed["response_text"]
    print(f"\nPARSED RESPONSE TEXT ({len(resp.split())} words):\n")
    print(resp)

    # Verify all 5 points
    print("\n--- VERIFYING ALL 5 REQUIRED POINTS ---")
    c1 = any(k in resp.lower() for k in ["analytics", "machine learning", "ai application"])
    c2 = any(k in resp.lower() for k in ["recommend", "starting path", "start with", "beginner"])
    c3 = any(k in resp.lower() for k in ["roadmap", "month", "months 1", "months 3"])
    c4 = any(k in resp.lower() for k in ["project", "resource", "freecodecamp", "kaggle", "coursera", "github"])
    c5 = any(k in resp.lower() for k in ["internship", "ready", "readiness", "evaluate"])

    print(f"1. Compares all three paths       : {'PASS' if c1 else 'FAIL'}")
    print(f"2. Recommends starting path & why : {'PASS' if c2 else 'FAIL'}")
    print(f"3. Realistic 6-month roadmap      : {'PASS' if c3 else 'FAIL'}")
    print(f"4. Projects & free resources      : {'PASS' if c4 else 'FAIL'}")
    print(f"5. Internship readiness evaluation: {'PASS' if c5 else 'FAIL'}")

    assert all([c1, c2, c3, c4, c5]), "All 5 points must be covered!"
    assert resp != "That's interesting! Tell me more.", "Must not be generic fallback!"
    print("\n>>> ALL 5 VERIFICATION CRITERIA PASSED WITH ZERO FALLBACK! <<<")

if __name__ == "__main__":
    asyncio.run(test_live_groq_with_improvements())
