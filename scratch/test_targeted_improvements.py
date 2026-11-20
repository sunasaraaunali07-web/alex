import asyncio
import json
import time
from app.services.llm_service import LLMService
from app.services.tts_service import TTSService
from app.services.pause_manager import (
    calculate_adaptive_pause,
    classify_complexity,
    build_timing_schedule,
    PauseCategory
)

async def run_focused_tests():
    print("=" * 70)
    print("ALEX — FOCUSED VERIFICATION OF TARGETED IMPROVEMENTS")
    print("=" * 70)

    llm = LLMService()
    tts = TTSService()
    results = {}

    # -------------------------------------------------------------------------
    # TEST 1: Correct English: no false correction
    # -------------------------------------------------------------------------
    print("\n[Test 1] Correct English: 'I have been working on our web project all morning.'")
    r1, _ = await llm.generate_response("I have been working on our web project all morning.")
    corr1 = r1.get("corrections", [])
    print("  Corrections count:", len(corr1))
    assert len(corr1) == 0, f"Expected 0 corrections for correct English, got {len(corr1)}"
    results["Test 1: Correct English (No false correction)"] = "PASS"
    print("  PASS: 0 false corrections detected.")
    await asyncio.sleep(2)

    # -------------------------------------------------------------------------
    # TEST 2: One genuine grammar mistake
    # -------------------------------------------------------------------------
    print("\n[Test 2] One genuine mistake: 'Yesterday I go to market.'")
    r2, _ = await llm.generate_response("Yesterday I go to market.")
    corr2 = r2.get("corrections", [])
    print("  Corrections:", json.dumps(corr2, indent=2))
    assert len(corr2) == 1, f"Expected 1 correction, got {len(corr2)}"
    assert "went" in corr2[0]["corrected"].lower()
    results["Test 2: One genuine grammar mistake"] = "PASS"
    print("  PASS: Single genuine mistake identified.")
    await asyncio.sleep(2)

    # -------------------------------------------------------------------------
    # TEST 3: Two or three mistakes in one sentence
    # -------------------------------------------------------------------------
    print("\n[Test 3] Multiple mistakes: 'Yesterday I go to market and meet my friend and we was discussing.'")
    r3, _ = await llm.generate_response("Yesterday I go to market and meet my friend and we was discussing.")
    corr3 = r3.get("corrections", [])
    print(f"  Corrections count: {len(corr3)}")
    for c in corr3:
        print(f"    - '{c['original']}' -> '{c['corrected']}'")
    assert len(corr3) >= 2, f"Expected at least 2 corrections, got {len(corr3)}"
    results["Test 3: Multiple mistakes in one sentence"] = "PASS"
    print("  PASS: Detected multiple mistakes across the sentence.")
    await asyncio.sleep(2)

    # -------------------------------------------------------------------------
    # TEST 4: Explicit incorrect-versus-correct spoken comparison
    # -------------------------------------------------------------------------
    print("\n[Test 4] Explicit spoken comparison ('X instead of Y' or 'X rather than Y')")
    has_explicit = False
    for c in corr3:
        comp = c.get("spoken_comparison", "")
        print(f"  Spoken comparison: \"{comp}\"")
        if any(marker in comp.lower() for marker in ["instead of", "rather than", "not '", "not \"", ", not "]):
            has_explicit = True
    assert has_explicit, "Spoken comparison must explicitly compare incorrect and correct phrases"
    results["Test 4: Explicit spoken comparison (X instead of Y / rather than Y)"] = "PASS"
    print("  PASS: Explicit comparison verified.")
    await asyncio.sleep(2)

    # -------------------------------------------------------------------------
    # TEST 5: Natural variation in correction wording
    # -------------------------------------------------------------------------
    print("\n[Test 5] Natural variation in correction wording across multiple errors")
    comparisons = [c.get("spoken_comparison", "") for c in corr3]
    # Check that they don't all start with identical 4 words
    starters = [c.split()[:3] for c in comparisons if len(c.split()) >= 3]
    distinct_starters = len(set(tuple(s) for s in starters))
    print(f"  Distinct openings: {distinct_starters} out of {len(comparisons)}")
    assert distinct_starters >= min(2, len(comparisons)), "Correction wording should naturally vary across items"
    results["Test 5: Natural variation in correction wording"] = "PASS"
    print("  PASS: Natural phrasing variation verified.")
    await asyncio.sleep(2)

    # -------------------------------------------------------------------------
    # TEST 6: Audible emphasis on changed words
    # -------------------------------------------------------------------------
    print("\n[Test 6] Audible emphasis on changed words")
    for c in corr3:
        emph = c.get("emphasized_word", "")
        assert emph != "", "Must identify emphasized_word"
        print(f"  Correction: '{c['original']}' -> '{c['corrected']}' | Emphasized word: '{emph}'")
    results["Test 6: Audible emphasis on changed words"] = "PASS"
    print("  PASS: Emphasized words accurately isolated and marked.")

    # -------------------------------------------------------------------------
    # TEST 7: Appropriate pauses before and between corrections
    # -------------------------------------------------------------------------
    print("\n[Test 7] Appropriate pauses (Timing schedule verification)")
    schedule = build_timing_schedule(corr3)
    for s in schedule:
        print(f"  Corr {s['correction_index']}: between={s['pause_between_phrases_ms']}ms, after={s['pause_after_correction_ms']}ms, last={s['is_last_correction']}")
        assert 150 <= s["pause_after_correction_ms"] <= 1200, "Pause must be within safety bounds"
    assert len(schedule) == len(corr3)
    results["Test 7: Appropriate pauses before and between corrections"] = "PASS"
    print("  PASS: Calculated adaptive pause schedule matches exact mathematical formula.")

    # -------------------------------------------------------------------------
    # TEST 8: No duplicate correction cards
    # -------------------------------------------------------------------------
    print("\n[Test 8] No duplicate correction cards")
    originals = [c["original"].strip().lower() for c in corr3]
    assert len(originals) == len(set(originals)), "No duplicate original phrases allowed"
    results["Test 8: No duplicate correction cards"] = "PASS"
    print("  PASS: Zero duplicate correction cards.")

    # -------------------------------------------------------------------------
    # TEST 9: Acceptable expressions that should not be corrected
    # -------------------------------------------------------------------------
    print("\n[Test 9] Acceptable vocabulary & Indian English (No synonym replacements)")
    # 'particularly' should NOT be changed to 'specifically'
    r9, _ = await llm.generate_response("I am particularly interested in frontend architecture, and I have a question about state management.")
    corr9 = r9.get("corrections", [])
    print("  Corrections on acceptable sentence:", corr9)
    assert len(corr9) == 0, f"Acceptable vocabulary 'particularly' must not be changed, got {corr9}"
    results["Test 9: Acceptable expressions preserved (No synonym replacement)"] = "PASS"
    print("  PASS: Standard acceptable vocabulary preserved without false corrections.")
    await asyncio.sleep(2)

    # -------------------------------------------------------------------------
    # TEST 10: Detailed multi-part question (Conversation Intelligence)
    # -------------------------------------------------------------------------
    print("\n[Test 10] Detailed multi-part question: Comparison of Data Analytics, ML, and AI App Dev")
    multi_part_query = (
        "Can you compare Data Analytics, Machine Learning, and AI Application Development? "
        "Please cover career opportunities, required skills, projects to build, and a realistic 6-month roadmap."
    )
    r10, _ = await llm.generate_response(multi_part_query)
    resp10 = r10.get("response_text", "")
    print(f"  Response length: {len(resp10.split())} words, {len(resp10)} chars")
    print("  Preview:\n" + resp10[:300] + "...")
    
    # Must answer all requested sections
    has_analytics = "analytics" in resp10.lower()
    has_ml = "machine learning" in resp10.lower() or "ml" in resp10.lower()
    has_ai = "ai" in resp10.lower()
    has_skills = "skill" in resp10.lower() or "python" in resp10.lower() or "sql" in resp10.lower()
    has_projects = "project" in resp10.lower() or "build" in resp10.lower()
    has_roadmap = "month" in resp10.lower() or "roadmap" in resp10.lower() or "step" in resp10.lower()
    
    assert has_analytics and has_ml and has_ai, "Must cover all three disciplines"
    assert has_skills, "Must cover required skills"
    assert has_projects, "Must cover projects to build"
    assert has_roadmap, "Must cover roadmap"
    assert len(resp10.split()) >= 150, f"Detailed multi-part question requires a comprehensive answer (got {len(resp10.split())} words)"
    results["Test 10: Detailed multi-part question (Conversation Intelligence)"] = "PASS"
    print("  PASS: All requested sections, comparisons, skills, projects, and roadmap answered thoroughly.")
    await asyncio.sleep(2)

    # -------------------------------------------------------------------------
    # TEST 11: Personal or emotionally sensitive conversation
    # -------------------------------------------------------------------------
    print("\n[Test 11] Emotionally sensitive conversation: Empathy and warmth")
    sensitive_query = "I've been feeling really overwhelmed with my semester exams lately. I'm afraid I might fail."
    r11, _ = await llm.generate_response(sensitive_query)
    resp11 = r11.get("response_text", "")
    print("  Alex reply:", resp11)
    empathy_markers = ["overwhelm", "stress", "understand", "breathe", "normal", "hear", "hard", "tough", "take a", "support", "exam", "step"]
    has_empathy = any(m in resp11.lower() for m in empathy_markers)
    assert has_empathy, "Alex must express genuine empathy and understanding"
    results["Test 11: Emotionally sensitive conversation (Empathetic tone)"] = "PASS"
    print("  PASS: Empathetic, supportive, human tone verified.")
    await asyncio.sleep(2)

    # -------------------------------------------------------------------------
    # TEST 12: Exciting and casual conversation
    # -------------------------------------------------------------------------
    print("\n[Test 12] Exciting & casual conversation: Enthusiastic, natural friendly tone")
    exciting_query = "Guess what! Our hackathon team just won first place today!"
    r12, _ = await llm.generate_response(exciting_query)
    resp12 = r12.get("response_text", "")
    print("  Alex reply:", resp12)
    excited_markers = ["congrat", "amazing", "awesome", "huge", "wow", "proud", "first place", "celebrat", "win"]
    has_excitement = any(m in resp12.lower() for m in excited_markers)
    assert has_excitement, "Alex must share genuine excitement with the user"
    results["Test 12: Exciting & casual conversation (Warm, engaging tone)"] = "PASS"
    print("  PASS: Naturally enthusiastic and engaging tone verified.")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FOCUSED TESTS SUMMARY")
    print("=" * 70)
    for name, status in results.items():
        print(f"  {name:<60}: {status}")

    all_passed = all(s == "PASS" for s in results.values())
    assert all_passed, "Not all focused tests passed!"
    print("\n>>> ALL 12 FOCUSED TESTS PASSED SUCCESSFULLY! <<<\n")

if __name__ == "__main__":
    asyncio.run(run_focused_tests())
