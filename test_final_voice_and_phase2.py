import asyncio
import json
import re
import sys
from app.services.llm_service import LLMService, _phrase_exists_in_user_text
from app.services.tts_service import TTSService
from app.services.pause_manager import (
    calculate_adaptive_pause,
    classify_complexity,
    build_timing_schedule,
    PauseCategory,
    MIN_PAUSE_MS,
    MAX_PAUSE_MS
)
from app.services.session_manager import session_manager
from app.services.analysis_service import analysis_service

async def run_final_verification():
    print("=" * 70)
    print("ALEX — FINAL VOICE EXPERIENCE & PHASE 2 VERIFICATION SUITE")
    print("=" * 70)

    llm = LLMService()
    tts = TTSService()
    test_results = {}

    # -------------------------------------------------------------------------
    # TEST 1: Correct English (No false corrections)
    # -------------------------------------------------------------------------
    print("\n[Test 1] Correct English: 'I have been studying computer science for three years.'")
    res1, _ = await llm.generate_response("I have been studying computer science for three years.")
    print("  Result 1 corrections count:", len(res1.get("corrections", [])))
    assert len(res1.get("corrections", [])) == 0, "Correct English must receive 0 corrections"
    test_results["Test 1: Correct English (No false corrections)"] = "PASS"
    print("  PASS: 0 false corrections detected.")
    await asyncio.sleep(2)

    # -------------------------------------------------------------------------
    # TEST 2: Simple mistake ("Yesterday I go" -> "went")
    # -------------------------------------------------------------------------
    print("\n[Test 2] Simple mistake: 'Yesterday I go to college and met my professor.'")
    res2, _ = await llm.generate_response("Yesterday I go to college and met my professor.")
    corr2 = res2.get("corrections", [])
    print("  Corrections:", json.dumps(corr2, indent=2))
    assert len(corr2) >= 1, "Must detect the tense mistake"
    c2 = corr2[0]
    assert "went" in c2["corrected"].lower(), "Must correct 'go' to 'went'"
    assert c2.get("emphasized_word"), "Must provide emphasized_word"
    assert c2.get("spoken_original"), "Must provide spoken_original"
    assert c2.get("spoken_corrected"), "Must provide spoken_corrected"
    assert c2.get("full_original_sentence"), "Must provide full_original_sentence"
    assert c2.get("full_corrected_sentence"), "Must provide full_corrected_sentence"
    test_results["Test 2: Simple mistake (Phrase-level correction)"] = "PASS"
    print("  PASS: Detected mistake, produced phrase-level spoken correction and full card sentences.")
    await asyncio.sleep(2)

    # -------------------------------------------------------------------------
    # TEST 3: Multiple mistakes (Phrase-level speech, NO spoken explanations)
    # -------------------------------------------------------------------------
    print("\n[Test 3] Multiple mistakes: 'He don't know the answer and he say nothing.'")
    res3, _ = await llm.generate_response("He don't know the answer and he say nothing.")
    corr3 = res3.get("corrections", [])
    spoken3 = res3.get("spoken_text", "")
    print("  Corrections count:", len(corr3))
    print("  Spoken text:", spoken3)
    assert len(corr3) >= 2, "Must detect both 'don't' and 'say'"
    for c in corr3:
        # Crucial check: Explanation must NOT be in spoken text
        expl = c.get("explanation", "").strip()
        if expl and len(expl) > 10:
            assert expl not in spoken3, f"Explanation '{expl}' must NOT be spoken aloud!"
    test_results["Test 3: Multiple mistakes (Explanations strictly written)"] = "PASS"
    print("  PASS: Spoken text contains only phrase corrections, explanations strictly kept to written cards.")
    await asyncio.sleep(2)

    # -------------------------------------------------------------------------
    # TEST 4: Natural expression ("out of my hand" -> expression type)
    # -------------------------------------------------------------------------
    print("\n[Test 4] Natural expression: 'Everything is going out of my hand.'")
    res4, _ = await llm.generate_response("Everything is going out of my hand.")
    corr4 = res4.get("corrections", [])
    print("  Corrections:", json.dumps(corr4, indent=2))
    assert len(corr4) >= 1, "Must detect unnatural idiom"
    assert any(c.get("type") == "expression" for c in corr4), "Must categorize idiom as type 'expression'"
    test_results["Test 4: Natural expression (Categorized as expression)"] = "PASS"
    print("  PASS: Correctly categorized idiom as type 'expression'.")
    await asyncio.sleep(2)

    # -------------------------------------------------------------------------
    # TEST 5: Previous-turn contamination (Turn 1 mistake -> Turn 2 clean: 0)
    # -------------------------------------------------------------------------
    print("\n[Test 5] Previous-turn contamination check across multi-turn dialogue")
    sid5 = "test-final-contamination"
    session_manager.get_or_create(sid5)
    t1_msg = "Yesterday I go to market."
    t1_res, _ = await llm.generate_response(t1_msg)
    session_manager.add_turn(sid5, t1_msg, t1_res["response_text"], corrections=t1_res["corrections"])
    await asyncio.sleep(2)

    hist5 = session_manager.get_dialogue_history_for_llm(sid5)
    t2_clean = "The weather in my city has been remarkably pleasant all week."
    t2_res, _ = await llm.generate_response(t2_clean, history=hist5)
    print("  Turn 2 corrections count:", len(t2_res.get("corrections", [])))
    assert len(t2_res.get("corrections", [])) == 0, "Turn 2 clean sentence must have 0 corrections"
    test_results["Test 5: Previous-turn contamination (Zero contamination)"] = "PASS"
    print("  PASS: Zero contamination from previous turn into current turn.")
    await asyncio.sleep(2)

    # -------------------------------------------------------------------------
    # TEST 6: Unclear audio (ASR uncertainty protection)
    # -------------------------------------------------------------------------
    print("\n[Test 6] Unclear audio (ASR uncertainty protection)")
    asr_meta = {
        "avg_logprob": -1.3,
        "no_speech_prob": 0.65,
        "is_globally_uncertain": True
    }
    res6, _ = await llm.generate_response("I go to store yesterday maybe", asr_metadata=asr_meta)
    print("  Uncertain turn corrections:", res6.get("corrections", []))
    assert len(res6.get("corrections", [])) == 0, "Corrections must be suppressed on uncertain audio"
    test_results["Test 6: ASR uncertainty protection (Suppression verified)"] = "PASS"
    print("  PASS: Corrections suppressed deterministically when ASR confidence is low.")
    await asyncio.sleep(2)

    # -------------------------------------------------------------------------
    # TEST 7: Natural conversation (Grammatical error + question)
    # -------------------------------------------------------------------------
    print("\n[Test 7] Natural conversation: 'Yesterday I go to cinema. What movies you like?'")
    res7, _ = await llm.generate_response("Yesterday I go to cinema. What movies you like?")
    print("  Response text:", res7.get("response_text"))
    print("  Spoken text:", res7.get("spoken_text"))
    print("  Corrections count:", len(res7.get("corrections", [])))
    assert len(res7.get("corrections", [])) >= 1, "Must correct grammar"
    assert len(res7.get("response_text", "")) > 10, "Must provide conversational reply"
    assert "?" in res7.get("response_text", "") or "movie" in res7.get("response_text", "").lower(), "Must answer/continue the conversation topic"
    test_results["Test 7: Natural conversation (Corrected + Conversational)"] = "PASS"
    print("  PASS: Addressed both correction and conversational dialogue naturally.")
    await asyncio.sleep(2)

    # -------------------------------------------------------------------------
    # TEST 8: Multiple correction timing (Verify P = B + (C * K) + (M * D))
    # -------------------------------------------------------------------------
    print("\n[Test 8] Multiple correction timing formula verification")
    # P = B + (C * K) + (M * D)
    # Baseline between phrases: B=400, C=1, K=100, M=0, D=100 => P=500ms
    p_phrase = calculate_adaptive_pause(PauseCategory.AFTER_CORRECTION_PHRASE, complexity=1, correction_index=0, custom_baseline=400)
    assert p_phrase == 500, f"Expected 500ms, got {p_phrase}ms"
    
    # Baseline after correction: B=650, C=2, K=100, M=1, D=100 => P = 650 + 200 + 100 = 950ms
    p_after_corr1 = calculate_adaptive_pause(PauseCategory.BETWEEN_CORRECTIONS, complexity=2, correction_index=1, custom_baseline=650)
    assert p_after_corr1 == 950, f"Expected 950ms, got {p_after_corr1}ms"

    # User example from prompt: B=400, C=2, K=100, M=1, D=100 => P = 400 + 200 + 100 = 700ms
    p_user_ex = calculate_adaptive_pause(PauseCategory.BETWEEN_CORRECTIONS, complexity=2, correction_index=1, custom_baseline=400)
    assert p_user_ex == 700, f"Expected 700ms, got {p_user_ex}ms"

    test_results["Test 8: Multiple correction timing formula (P = B + (C*K) + (M*D))"] = "PASS"
    print("  PASS: Exact mathematical formula match verified.")

    # -------------------------------------------------------------------------
    # TEST 9: Correction complexity (Verify Level 1 vs Level 3 timing)
    # -------------------------------------------------------------------------
    print("\n[Test 9] Correction complexity classification & timing")
    c_lev1 = classify_complexity("I go", "I went", "grammar", "Past tense")
    c_lev3 = classify_complexity(
        "If I was know that you was coming yesterday I would be waiting",
        "Had I known you were coming yesterday, I would have waited for you",
        "grammar",
        "Past unreal conditional requires inverted pluperfect subjunctive and perfect conditional"
    )
    assert c_lev1 == 1, f"Expected Level 1, got {c_lev1}"
    assert c_lev3 == 3, f"Expected Level 3, got {c_lev3}"

    p_lev1 = calculate_adaptive_pause(PauseCategory.BETWEEN_CORRECTIONS, complexity=c_lev1, correction_index=0)
    p_lev3 = calculate_adaptive_pause(PauseCategory.BETWEEN_CORRECTIONS, complexity=c_lev3, correction_index=0)
    print(f"  Level 1 pause: {p_lev1}ms, Level 3 pause: {p_lev3}ms")
    assert p_lev3 > p_lev1, "Higher complexity must produce longer cognitive pause"
    assert p_lev3 - p_lev1 == 200, "Difference between Level 1 and Level 3 must be 2 * 100ms = 200ms"
    test_results["Test 9: Correction complexity (Levels 1 & 3 timing)"] = "PASS"
    print("  PASS: Complexity classification and dynamic pause scaling verified.")

    # -------------------------------------------------------------------------
    # TEST 10: Segmented TTS Synthesis & Segment Generation
    # -------------------------------------------------------------------------
    print("\n[Test 10] Segmented TTS synthesis & speech segments verification")
    turn_tts = await tts.synthesize_turn(
        corrections=corr2,
        response_text=res2["response_text"],
        spoken_text=res2["spoken_text"]
    )
    segments = turn_tts.get("speech_segments", [])
    print(f"  Synthesized {len(segments)} distinct speech segments:")
    for s in segments:
        print(f"    - Type: {s['type']}, Text: '{s['text']}', Pause After: {s['pause_after_ms']}ms, Card: {s['card_index']}")
    
    assert len(segments) >= 2, "Must contain at least 1 correction segment + 1 reply segment"
    corr_segs = [s for s in segments if s["type"].startswith("corr")]
    reply_segs = [s for s in segments if s["type"] == "response"]
    assert len(corr_segs) >= 1, "Must contain correction segment"
    assert len(reply_segs) == 1, "Must contain response segment"
    assert all(s.get("audio_base64") for s in segments), "All segments must contain synthesized audio_base64"
    test_results["Test 10: Segmented TTS synthesis & speech segments"] = "PASS"
    print("  PASS: Segmented audio generation and timing metadata complete.")

    # -------------------------------------------------------------------------
    # TEST 11: Written Card Formatting & Emphasized Word Extraction
    # -------------------------------------------------------------------------
    print("\n[Test 11] Written card formatting & emphasized word extraction")
    for c in corr2:
        assert "full_original_sentence" in c, "Must include full original sentence"
        assert "full_corrected_sentence" in c, "Must include full corrected sentence"
        assert "emphasized_word" in c, "Must include emphasized word"
        assert "explanation" in c, "Must include explanation"
        assert c["full_original_sentence"] != "", "Original sentence must not be empty"
        assert c["full_corrected_sentence"] != "", "Corrected sentence must not be empty"
    test_results["Test 11: Written card structure (Full sentences + Emphasis + Why)"] = "PASS"
    print("  PASS: Full original, full corrected, emphasized word, and explanation verified.")

    # -------------------------------------------------------------------------
    # TEST 12: Long conversational turn (Multi-sentence paragraph analysis)
    # -------------------------------------------------------------------------
    print("\n[Test 12] Long conversational turn analysis")
    long_turn = (
        "Last night I was watching movie with my cousin. "
        "He don't like horror movies so he was getting scared. "
        "We was eating popcorn and having good time."
    )
    res12, _ = await llm.generate_response(long_turn)
    corr12 = res12.get("corrections", [])
    print(f"  Detected {len(corr12)} errors in multi-sentence paragraph:")
    for c in corr12:
        print(f"    - '{c['original']}' -> '{c['corrected']}' (Spoken: '{c['spoken_original']}' -> '{c['spoken_corrected']}')")
    assert len(corr12) >= 2, "Must detect multiple distributed mistakes in long turn"
    test_results["Test 12: Long conversational turn (Multi-sentence analysis)"] = "PASS"
    print("  PASS: Comprehensive multi-sentence detection verified.")

    # -------------------------------------------------------------------------
    # FINAL RESULTS SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 70)
    for name, status in test_results.items():
        print(f"  {name:<60}: {status}")

    all_passed = all(s == "PASS" for s in test_results.values())
    assert all_passed, "Not all tests passed!"
    print("\n>>> ALL 12 FINAL VERIFICATION TESTS PASSED SUCCESSFULLY! <<<\n")

if __name__ == "__main__":
    asyncio.run(run_final_verification())
