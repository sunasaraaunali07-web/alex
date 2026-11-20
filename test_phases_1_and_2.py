import asyncio
import json
import time
from app.services.llm_service import LLMService, _parse_llm_json
from app.services.session_manager import session_manager
from app.services.analysis_service import analysis_service

async def run_tests():
    print("=" * 60)
    print("ALEX — PHASE 1 & PHASE 2 VERIFICATION SUITE")
    print("=" * 60)
    
    llm = LLMService()
    test_results = {}

    # -------------------------------------------------------------
    # PHASE 1 TESTS
    # -------------------------------------------------------------
    print("\n--- PHASE 1: Real-Time Grammar Correction & Dialogue ---")

    # TEST 1: Basic grammar
    print("\n[Test 1] Basic grammar: 'Yesterday I go to market and buy some clothes.'")
    res1, ms1 = await llm.generate_response("Yesterday I go to market and buy some clothes.")
    print("Result 1:", json.dumps(res1, indent=2))
    corr1 = res1.get("corrections", [])
    has_went = any("went" in c.get("corrected", "").lower() for c in corr1) or "went" in res1.get("spoken_text", "").lower()
    has_bought = any("bought" in c.get("corrected", "").lower() for c in corr1) or "bought" in res1.get("spoken_text", "").lower()
    test1_pass = len(corr1) >= 1 and (has_went or has_bought)
    test_results["Test 1 (Basic grammar)"] = "PASS" if test1_pass else "FAIL"
    print(f"Test 1: {'PASS' if test1_pass else 'FAIL'}")
    await asyncio.sleep(2)

    # TEST 2: Correct sentence protection
    print("\n[Test 2] Correct sentence: 'He can help me.'")
    res2, ms2 = await llm.generate_response("He can help me.")
    print("Result 2:", json.dumps(res2, indent=2))
    corr2 = res2.get("corrections", [])
    test2_pass = len(corr2) == 0
    test_results["Test 2 (Correct sentence protection)"] = "PASS" if test2_pass else "FAIL"
    print(f"Test 2: {'PASS' if test2_pass else 'FAIL'}")
    await asyncio.sleep(2)

    # TEST 3: Natural expression vs grammar
    print("\n[Test 3] Natural expression: 'Everything is going out of my hand.'")
    res3, ms3 = await llm.generate_response("Everything is going out of my hand.")
    print("Result 3:", json.dumps(res3, indent=2))
    corr3 = res3.get("corrections", [])
    # Should suggest 'control' or similar natural phrasing, preferably tagged 'expression'
    has_control_or_expression = any("control" in c.get("corrected", "").lower() or c.get("type") == "expression" for c in corr3) or "control" in res3.get("spoken_text", "").lower()
    test3_pass = len(corr3) >= 1 and has_control_or_expression
    test_results["Test 3 (Natural expression)"] = "PASS" if test3_pass else "FAIL"
    print(f"Test 3: {'PASS' if test3_pass else 'FAIL'}")
    await asyncio.sleep(2)

    # TEST 4: Multiple mistakes
    print("\n[Test 4] Multiple mistakes: 'Today I was discussing with my friend about our project. He don't understand how the backend works. I explained him everything but he still have some confusion.'")
    long_msg = "Today I was discussing with my friend about our project. He don't understand how the backend works. I explained him everything but he still have some confusion."
    res4, ms4 = await llm.generate_response(long_msg)
    print("Result 4:", json.dumps(res4, indent=2))
    corr4 = res4.get("corrections", [])
    has_doesnt = any("doesn't" in c.get("corrected", "").lower() or "does not" in c.get("corrected", "").lower() for c in corr4)
    has_has = any("has" in c.get("corrected", "").lower() for c in corr4)
    test4_pass = len(corr4) >= 2 and (has_doesnt or has_has)
    test_results["Test 4 (Multiple mistakes)"] = "PASS" if test4_pass else "FAIL"
    print(f"Test 4: {'PASS' if test4_pass else 'FAIL'}")
    await asyncio.sleep(2)

    # TEST 5: ASR uncertainty protection
    print("\n[Test 5] ASR uncertainty: Low confidence metadata with homophone ambiguity")
    asr_meta_low = {"avg_logprob": -1.2, "no_speech_prob": 0.55}
    res5, ms5 = await llm.generate_response("Yesterday I go there", asr_metadata=asr_meta_low)
    print("Result 5 (Low ASR confidence):", json.dumps(res5, indent=2))
    corr5 = res5.get("corrections", [])
    test5_pass = len(corr5) == 0  # Under uncertain audio, corrections should be suppressed
    test_results["Test 5 (ASR uncertainty protection)"] = "PASS" if test5_pass else "FAIL"
    print(f"Test 5: {'PASS' if test5_pass else 'FAIL'}")
    await asyncio.sleep(2)

    # TEST 6: Short conversational phrase
    print("\n[Test 6] Short conversational phrase: 'I still not completed mine.'")
    res6, ms6 = await llm.generate_response("I still not completed mine.")
    print("Result 6:", json.dumps(res6, indent=2))
    corr6 = res6.get("corrections", [])
    has_havent = any("haven't" in c.get("corrected", "").lower() or "have not" in c.get("corrected", "").lower() for c in corr6) or "haven't" in res6.get("spoken_text", "").lower()
    test6_pass = len(corr6) >= 1 and has_havent
    test_results["Test 6 (Short conversational phrase)"] = "PASS" if test6_pass else "FAIL"
    print(f"Test 6: {'PASS' if test6_pass else 'FAIL'}")
    await asyncio.sleep(2)

    # TEST 7: No correction overload
    print("\n[Test 7] No correction overload on complex sentence")
    test7_pass = len(corr4) <= 5  # Step 2: Uncapped detection of genuine mistakes without excessive hallucination
    test_results["Test 7 (No correction overload)"] = "PASS" if test7_pass else "FAIL"
    print(f"Test 7 (Count was {len(corr4)} <= 5): {'PASS' if test7_pass else 'FAIL'}")
    await asyncio.sleep(2)

    # TEST 8: Memory & context recall
    print("\n[Test 8] Multi-turn memory recall across conversational turns")
    sid_mem = "test-session-memory"
    session_manager.get_or_create(sid_mem)
    # Turn 1: user introduces personal fact
    t1_u = "My name is Rahul. I live in Pune and I am studying computer engineering."
    t1_a, _ = await llm.generate_response(t1_u)
    session_manager.add_turn(sid_mem, t1_u, t1_a["response_text"], corrections=t1_a["corrections"])
    await asyncio.sleep(2)
    
    # Turn 2: casual filler
    t2_u = "The weather is really nice today."
    h2 = session_manager.get_dialogue_history_for_llm(sid_mem, window=30)
    t2_a, _ = await llm.generate_response(t2_u, history=h2)
    session_manager.add_turn(sid_mem, t2_u, t2_a["response_text"], corrections=t2_a["corrections"])
    await asyncio.sleep(2)

    # Turn 3: Ask about earlier fact
    t3_u = "Do you remember which city I live in and what I study?"
    h3 = session_manager.get_dialogue_history_for_llm(sid_mem, window=30)
    t3_a, _ = await llm.generate_response(t3_u, history=h3)
    session_manager.add_turn(sid_mem, t3_u, t3_a["response_text"], corrections=t3_a["corrections"])
    await asyncio.sleep(2)
    
    resp_text_3 = t3_a["response_text"].lower()
    has_pune = "pune" in resp_text_3
    has_eng = "engineering" in resp_text_3 or "computer" in resp_text_3
    test8_pass = has_pune and has_eng
    print("Memory response:", t3_a["response_text"])
    test_results["Test 8 (Conversation memory)"] = "PASS" if test8_pass else "FAIL"
    print(f"Test 8: {'PASS' if test8_pass else 'FAIL'}")

    # TEST 9: Session isolation
    print("\n[Test 9] Session isolation check")
    sid_iso_a = "test-session-a"
    sid_iso_b = "test-session-b"
    session_manager.get_or_create(sid_iso_a)
    session_manager.get_or_create(sid_iso_b)
    session_manager.add_turn(sid_iso_a, "User in A", "Alex in A")
    session_manager.add_turn(sid_iso_b, "User in B", "Alex in B")
    hist_a = session_manager.get_complete_transcript(sid_iso_a)
    hist_b = session_manager.get_complete_transcript(sid_iso_b)
    test9_pass = len(hist_a) == 2 and len(hist_b) == 2 and hist_a[0]["content"] != hist_b[0]["content"]
    test_results["Test 9 (Session isolation)"] = "PASS" if test9_pass else "FAIL"
    print(f"Test 9: {'PASS' if test9_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # PHASE 2 TESTS
    # -------------------------------------------------------------
    print("\n--- PHASE 2: Intelligent Feedback & Learning Engine ---")

    # TEST 10: Recurring mistakes identification
    print("\n[Test 10] Recurring mistakes pattern identification")
    sid_rec = "test-session-recurring"
    session_manager.get_or_create(sid_rec)
    session_manager.add_turn(sid_rec, "Yesterday I go to college with my friend.", "That sounds fun! What classes did you take?")
    session_manager.add_turn(sid_rec, "Last week I meet him at the library.", "Nice! Did you study together?")
    session_manager.add_turn(sid_rec, "Two days ago we eat lunch together.", "What did you have for lunch?")
    
    rec_transcript = session_manager.get_complete_transcript(sid_rec)
    report10 = await analysis_service.analyze_session(rec_transcript)
    print("Report 10:", json.dumps(report10, indent=2))
    recurring_list = report10.get("recurring_mistakes", [])
    has_past_tense_pattern = any("past" in r.get("category", "").lower() or "tense" in r.get("category", "").lower() or "verb" in r.get("category", "").lower() for r in recurring_list)
    test10_pass = len(recurring_list) >= 1 and has_past_tense_pattern
    test_results["Test 10 (Recurring mistakes)"] = "PASS" if test10_pass else "FAIL"
    print(f"Test 10: {'PASS' if test10_pass else 'FAIL'}")

    # TEST 11: Isolated mistake (must not become a recurring pattern)
    print("\n[Test 11] Isolated mistake: exactly one error in an otherwise clean conversation")
    sid_iso = "test-session-isolated"
    session_manager.get_or_create(sid_iso)
    session_manager.add_turn(sid_iso, "I really enjoy reading science fiction books.", "Science fiction is fascinating! What is your favorite book?")
    session_manager.add_turn(sid_iso, "Yesterday I go to library to find a new one.", "Did you find anything interesting?")
    session_manager.add_turn(sid_iso, "Yes, I found a great book about space exploration.", "Space exploration stories are always inspiring.")

    iso_transcript = session_manager.get_complete_transcript(sid_iso)
    report11 = await analysis_service.analyze_session(iso_transcript)
    print("Report 11:", json.dumps(report11, indent=2))
    rec11 = report11.get("recurring_mistakes", [])
    # An isolated single error should NOT be counted as a recurring pattern (requires 2+ occurrences)
    test11_pass = len(rec11) == 0
    test_results["Test 11 (Isolated mistake protection)"] = "PASS" if test11_pass else "FAIL"
    print(f"Test 11: {'PASS' if test11_pass else 'FAIL'}")

    # TEST 12: Correct-English protection in final report
    print("\n[Test 12] Correct-English protection in final report")
    sid_clean = "test-session-clean"
    session_manager.get_or_create(sid_clean)
    session_manager.add_turn(sid_clean, "I have been working as a software developer for three years.", "That is great! What tech stack do you usually use?")
    session_manager.add_turn(sid_clean, "I mainly build web applications using modern frameworks.", "Modern web frameworks make development very efficient.")
    
    clean_transcript = session_manager.get_complete_transcript(sid_clean)
    report12 = await analysis_service.analyze_session(clean_transcript)
    print("Report 12:", json.dumps(report12, indent=2))
    rec12 = report12.get("recurring_mistakes", [])
    test12_pass = len(rec12) == 0
    test_results["Test 12 (Correct-English protection in report)"] = "PASS" if test12_pass else "FAIL"
    print(f"Test 12: {'PASS' if test12_pass else 'FAIL'}")

    # TEST 13: End-of-conversation report state machine & HTTP 202 flow
    print("\n[Test 13] End-of-conversation state machine")
    sid_end = "test-session-end"
    session_manager.get_or_create(sid_end)
    session_manager.add_turn(sid_end, "Hello Alex, good morning.", "Good morning! How are you doing today?")
    session_manager.add_turn(sid_end, "I am doing well, thank you.", "Glad to hear that! What are your plans for the day?")
    can_start, status_str, s_obj = session_manager.try_start_analysis(sid_end)
    test13_pass = can_start is True and status_str == "analyzing" and s_obj.status == "analyzing"
    test_results["Test 13 (End-of-conversation trigger & state)"] = "PASS" if test13_pass else "FAIL"
    print(f"Test 13: {'PASS' if test13_pass else 'FAIL'}")

    # TEST 14: Original transcript integrity
    print("\n[Test 14] Original transcript integrity check")
    orig_text = "Yesterday I go to market"
    corr_item = [{"original": "Yesterday I go to market", "corrected": "Yesterday I went to market", "explanation": "past tense", "type": "grammar"}]
    session_manager.add_turn(sid_end, orig_text, "Alex reply", corrections=corr_item)
    full_t = session_manager.get_complete_transcript(sid_end)
    last_user_turn = [m for m in full_t if m["role"] == "user"][-1]
    # Check that original text is unchanged and corrections are stored separately
    test14_pass = (last_user_turn["content"] == orig_text) and (len(last_user_turn.get("corrections", [])) == 1)
    test_results["Test 14 (Original transcript integrity)"] = "PASS" if test14_pass else "FAIL"
    print(f"Test 14: {'PASS' if test14_pass else 'FAIL'}")

    # TEST 15: Long-conversation analysis & chunking
    print("\n[Test 15] Long-conversation chunking (10 user turns > 8 limit)")
    sid_long = "test-session-long"
    session_manager.get_or_create(sid_long)
    for i in range(10):
        session_manager.add_turn(sid_long, f"This is user message number {i+1} talking about our work.", f"Alex reply number {i+1}.")
    long_transcript = session_manager.get_complete_transcript(sid_long)
    chunks = analysis_service._chunk_dialogue(long_transcript, max_user_turns_per_chunk=8)
    test15_pass = len(chunks) == 2 and len(chunks[0]) == 16 and len(chunks[1]) == 4
    test_results["Test 15 (Long-conversation chunking)"] = "PASS" if test15_pass else "FAIL"
    print(f"Test 15 (Chunks created: {len(chunks)}): {'PASS' if test15_pass else 'FAIL'}")

    # TEST 16: Duplicate End Call protection
    print("\n[Test 16] Duplicate End Call protection")
    can_start_again, status_again, _ = session_manager.try_start_analysis(sid_end)
    test16_pass = (can_start_again is False) and (status_again == "already_analyzing")
    test_results["Test 16 (Duplicate End Call protection)"] = "PASS" if test16_pass else "FAIL"
    print(f"Test 16: {'PASS' if test16_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE — RESULTS SUMMARY")
    print("=" * 60)
    all_passed = True
    for t_name, t_res in test_results.items():
        print(f"{t_name:<45}: {t_res}")
        if t_res != "PASS":
            all_passed = False
    
    print("\nOVERALL STATUS:", "ALL 16 TESTS PASSED" if all_passed else "SOME TESTS FAILED")

if __name__ == "__main__":
    asyncio.run(run_tests())
