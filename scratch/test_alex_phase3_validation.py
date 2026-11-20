import asyncio
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.services.llm_service import LLMService
from app.services.tts_service import TTSService
from app.services.memory_service import memory_service

async def run_validation():
    print("=== STARTING PHASE 3 PERFECT ALEX VALIDATION ===")
    llm = LLMService()
    tts = TTSService()

    # ---------------------------------------------------------
    # TEST CASE 1: The "because I have to" test case
    # User's complaint was:
    # "you meet my requirment but in this test first insted of ans say is clearly speak by alex but in second case it only say 'say ' not the insted of ans the text weitten in say"
    # ---------------------------------------------------------
    text1 = "Yesterday I go to college and because I have to submit my assignment."
    print(f"\n[TEST 1] User utterance:\n{text1}")
    
    res1, duration_ms = await llm.generate_response(text1)
    corrections1 = res1.get("corrections", [])
    response1 = res1.get("response_text", "")
    print(f"Alex Response: {response1}")
    print(f"Corrections count: {len(corrections1)}")
    for i, c in enumerate(corrections1):
        print(f"  Card {i+1}:")
        print(f"    Original: {c.get('original')}")
        print(f"    Corrected: {c.get('corrected')}")
        print(f"    Spoken Comparison: {c.get('spoken_comparison')}")
        print(f"    Explanation: {c.get('explanation')}")

    # Synthesize speech segments with TTS
    tts_result1 = await tts.synthesize_turn(
        response_text=response1,
        corrections=corrections1,
        spoken_text=res1.get("spoken_text", response1)
    )
    segments1 = tts_result1.get("speech_segments", [])
    print(f"TTS Speech Segments generated: {len(segments1)}")
    for s in segments1:
        print(f"  Segment ({s.get('type')}): '{s.get('text')}' [pause: {s.get('pause_after_ms')}ms]")
        # Verify that the spoken comparison NEVER says just 'say ' or truncates target words
        txt = s.get("text", "")
        if s.get("type") == "correction":
            assert len(txt.split()) >= 5, f"Spoken comparison text too short: '{txt}'"
            assert "instead of" in txt.lower() or "rather than" in txt.lower(), f"Spoken comparison missing comparison phrase: '{txt}'"
            assert not txt.lower().startswith("also, say.") and not txt.lower().startswith("say ."), f"Spoken comparison truncated: '{txt}'"
            # Verify no 'Why' explanation read into audio
            assert "use 'had to'" not in txt.lower() and "use past tense" not in txt.lower(), f"Grammar explanation leaked into audio: '{txt}'"

    print(">>> TEST 1 PASSED: Both spoken comparisons are complete, fully articulated, and contain no leaked grammar rules!")

    # ---------------------------------------------------------
    # TEST CASE 2: Multi-sentence narrative with "he don't understand" and "start arguing with"
    # ---------------------------------------------------------
    text2 = (
        "Yesterday I was going to college but suddenly my friend called me and said that he is waiting for me since 2 hours. "
        "I told him that I already reached there but actually I was still at home. After that he became angry because I didn't "
        "inform him earlier and I was not able to explain him what happened. I think he don't understand my situation because "
        "whenever I try to explain something he always interrupts me and start arguing with without listening completely."
    )
    print(f"\n[TEST 2] User multi-sentence utterance:\n{text2}")

    res2, duration_ms2 = await llm.generate_response(text2)
    corrections2 = res2.get("corrections", [])
    response2 = res2.get("response_text", "")
    print(f"Alex Response: {response2}")
    print(f"Corrections count: {len(corrections2)}")
    for i, c in enumerate(corrections2):
        print(f"  Card {i+1}:")
        print(f"    Original: {c.get('original')}")
        print(f"    Corrected: {c.get('corrected')}")
        print(f"    Spoken Comparison: {c.get('spoken_comparison')}")
        print(f"    Explanation: {c.get('explanation')}")

    # Check key mistakes detected
    all_origs = " ".join([c.get("original", "").lower() for c in corrections2])
    print(f"All captured error phrases: {all_origs}")
    assert any(k in all_origs for k in ["don't", "dont", "understand"]), "Expected 'don't understand' to be captured"
    assert any(k in all_origs for k in ["argu", "start"]), "Expected 'start arguing' to be captured"

    # Synthesize speech segments with TTS
    tts_result2 = await tts.synthesize_turn(
        response_text=response2,
        corrections=corrections2,
        spoken_text=res2.get("spoken_text", response2)
    )
    segments2 = tts_result2.get("speech_segments", [])
    print(f"TTS Speech Segments generated: {len(segments2)}")
    for s in segments2:
        print(f"  Segment ({s.get('type')}): '{s.get('text')}' [pause: {s.get('pause_after_ms')}ms]")
        txt = s.get("text", "")
        if s.get("type") == "correction":
            assert len(txt.split()) >= 5, f"Spoken comparison text too short: '{txt}'"
            assert "instead of" in txt.lower() or "rather than" in txt.lower(), f"Missing comparison words: '{txt}'"

    # Conversational response checks
    assert not any(bad in response2.lower() for bad in ["html", "css", "<div", "javascript", "code"]), "Alex must not talk about coding or HTML"
    assert not any(bad in response2.lower() for bad in ["remember to say", "in your grammar", "you should have said"]), "Alex response must not lecture on grammar"

    print(">>> TEST 2 PASSED: Multi-error extraction clean, natural conversation, no grammar lecturing!")

    # ---------------------------------------------------------
    # TEST CASE 3: Phase 3 Persistent Memory Check
    # ---------------------------------------------------------
    print("\n[TEST 3] Persistent SQLite Memory Verification")
    uid = "test_user_phase3_validation"
    sid = "session_phase3_val_001"
    
    # Simulate user sharing facts and mistakes
    memory_service.record_message(sid, uid, "user", "I'm a computer engineering student aiming to become a machine learning engineer.", 0)
    memory_service.extract_turn_memories(
        uid, sid,
        "I'm a computer engineering student aiming to become a machine learning engineer.",
        "That's an exciting path!",
        [{"original": "I go yesterday", "corrected": "I went yesterday", "type": "grammar", "explanation": "Past tense"}]
    )

    mem = memory_service.get_all_user_memory(uid)
    print("Memory Profile:", mem.get("profile"))
    print("Memory Facts:", mem.get("facts"))
    print("Grammar Memory:", mem.get("grammar_memory"))

    assert mem.get("facts"), "User facts must be stored in memory"
    assert mem.get("grammar_memory"), "Grammar memory must be stored in memory"

    context_str = memory_service.get_user_memory_context(uid)
    print(f"Retrieved Memory Context for Alex:\n{context_str}")
    assert "Machine Learning Engineer" in context_str or "Computer Engineering" in context_str

    print(">>> TEST 3 PASSED: SQLite memory correctly stores facts, grammar errors, and surfaces context across sessions!")
    print("\n=== ALL VALIDATION TESTS PASSED PERFECTLY ===")

if __name__ == "__main__":
    asyncio.run(run_validation())
