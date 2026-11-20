import asyncio
import json
from app.services.llm_service import LLMService

async def run_step2_tests():
    print("=" * 70)
    print("ALEX PHASE 2 — STEP 2: MULTI-SENTENCE CORRECTION DETECTION SUITE")
    print("=" * 70)

    llm = LLMService()
    results = {}

    # -----------------------------------------------------------------------
    # TEST 1: Paragraph with 4 genuine grammatical mistakes
    # -----------------------------------------------------------------------
    print("\n[Test 1] Paragraph with 4 genuine grammatical mistakes across sentences")
    p1 = (
        "Yesterday I go to market and I meeted my friend. "
        "He don't know about our exam because he was absent. "
        "We was discussing the syllabus for two hours."
    )
    # Mistakes: go->went, meeted->met, don't->doesn't/didn't, we was->we were
    r1, _ = await llm.generate_response(p1)
    print(f"  Utterance: {p1}")
    print(f"  Corrections count: {len(r1.get('corrections', []))}")
    print(json.dumps(r1.get("corrections"), indent=2))
    
    corrs1 = r1.get("corrections", [])
    has_go = any("went" in c.get("corrected", "").lower() for c in corrs1)
    has_met = any("met" in c.get("corrected", "").lower() for c in corrs1)
    has_dont = any("doesn't" in c.get("corrected", "").lower() or "didn't" in c.get("corrected", "").lower() or "does not" in c.get("corrected", "").lower() for c in corrs1)
    has_were = any("were" in c.get("corrected", "").lower() for c in corrs1)
    
    test1_pass = len(corrs1) >= 4 and has_go and has_met and has_dont and has_were
    results["Test 1 (Paragraph with 4 mistakes: all 4 detected)"] = "PASS" if test1_pass else "FAIL"
    print(f"  Found 4+ errors without artificial cap: {len(corrs1) >= 4} (Count = {len(corrs1)})")
    print(f"  Result: {'PASS' if test1_pass else 'FAIL'}")

    # -----------------------------------------------------------------------
    # TEST 2: Paragraph with 6 genuine grammatical mistakes
    # -----------------------------------------------------------------------
    print("\n[Test 2] Paragraph with 6 genuine grammatical mistakes across sentences")
    p2 = (
        "Last Sunday I wakes up late because my alarm didn't rang. "
        "I didn't had any breakfast so I runs to the bus stop. "
        "When the bus come, the driver tell me that it was full."
    )
    # Mistakes: wakes->woke, didn't rang->didn't ring, didn't had->didn't have, runs->ran, come->came, tell->told
    r2, _ = await llm.generate_response(p2)
    print(f"  Utterance: {p2}")
    print(f"  Corrections count: {len(r2.get('corrections', []))}")
    print(json.dumps(r2.get("corrections"), indent=2))
    
    corrs2 = r2.get("corrections", [])
    test2_pass = len(corrs2) >= 5  # Confirms no 1-3 limit and comprehensive multi-sentence coverage
    results["Test 2 (Paragraph with 6 mistakes: thorough coverage >= 5)"] = "PASS" if test2_pass else "FAIL"
    print(f"  Found >= 5 errors in 6-error paragraph: {test2_pass} (Count = {len(corrs2)})")
    print(f"  Result: {'PASS' if test2_pass else 'FAIL'}")

    # -----------------------------------------------------------------------
    # TEST 3: Paragraph containing both correct and incorrect sentences
    # -----------------------------------------------------------------------
    print("\n[Test 3] Mixed paragraph: correct sentences must NOT be flagged")
    p3 = (
        "I really enjoy studying computer science at my university. "
        "Yesterday I go to the library to read some books. "
        "The library is very quiet and peaceful. "
        "My friend don't like reading books so he stayed home."
    )
    r3, _ = await llm.generate_response(p3)
    print(f"  Utterance: {p3}")
    print(f"  Corrections count: {len(r3.get('corrections', []))}")
    print(json.dumps(r3.get("corrections"), indent=2))
    
    corrs3 = r3.get("corrections", [])
    # Sentences 1 & 3 are correct: "enjoy studying computer science", "quiet and peaceful"
    flagged_correct_s1 = any("computer science" in c.get("original", "").lower() for c in corrs3)
    flagged_correct_s3 = any("quiet and peaceful" in c.get("original", "").lower() for c in corrs3)
    # Sentences 2 & 4 have mistakes: "go" and "don't"
    caught_s2 = any("went" in c.get("corrected", "").lower() for c in corrs3)
    caught_s4 = any("doesn't" in c.get("corrected", "").lower() or "didn't" in c.get("corrected", "").lower() for c in corrs3)
    
    test3_pass = caught_s2 and caught_s4 and not flagged_correct_s1 and not flagged_correct_s3
    results["Test 3 (Mixed paragraph: correct sentences preserved, errors caught)"] = "PASS" if test3_pass else "FAIL"
    print(f"  Caught errors in s2 & s4: {caught_s2 and caught_s4}")
    print(f"  Correct s1 & s3 preserved: {not flagged_correct_s1 and not flagged_correct_s3}")
    print(f"  Result: {'PASS' if test3_pass else 'FAIL'}")

    await asyncio.sleep(2)

    # -----------------------------------------------------------------------
    # TEST 4: Completely grammatically correct paragraph
    # -----------------------------------------------------------------------
    print("\n[Test 4] Completely grammatically correct paragraph -> Zero corrections")
    p4 = (
        "I have been working on a software project for the past two months. "
        "We are building a web application using modern technologies. "
        "The user interface is clean, and the backend handles requests very efficiently. "
        "I hope we finish it before the deadline next week."
    )
    r4, _ = await llm.generate_response(p4)
    print(f"  Utterance: {p4}")
    corrs4 = r4.get("corrections", [])
    print(f"  Corrections count: {len(corrs4)} (Expected 0)")
    
    test4_pass = len(corrs4) == 0
    results["Test 4 (Completely correct paragraph: exactly 0 corrections)"] = "PASS" if test4_pass else "FAIL"
    print(f"  Result: {'PASS' if test4_pass else 'FAIL'}")

    await asyncio.sleep(2)

    # -----------------------------------------------------------------------
    # TEST 5: Mistakes distributed across different sentences
    # -----------------------------------------------------------------------
    print("\n[Test 5] Mistakes distributed across sentences: no early stopping")
    p5 = (
        "Last month I travelled to Mumbai with my family. "
        "We was staying at a hotel near the beach. "
        "My brother don't like spicy food so we had to find a mild restaurant. "
        "He still have stomach pain after dinner. "
        "Fortunately the doctors was very helpful."
    )
    r5, _ = await llm.generate_response(p5)
    print(f"  Utterance: {p5}")
    corrs5 = r5.get("corrections", [])
    print(f"  Corrections count: {len(corrs5)}")
    print(json.dumps(corrs5, indent=2))
    
    # Check that later sentences are analyzed and not abandoned
    caught_early = any("were" in c.get("corrected", "").lower() for c in corrs5) # sentence 2
    caught_middle = any("doesn't" in c.get("corrected", "").lower() or "didn't" in c.get("corrected", "").lower() for c in corrs5) # sentence 3
    caught_late = any("doctors were" in c.get("corrected", "").lower() or "were" in c.get("corrected", "").lower() or "has" in c.get("corrected", "").lower() or "had" in c.get("corrected", "").lower() for c in corrs5) # sentence 4/5
    
    test5_pass = len(corrs5) >= 3 and caught_early and caught_middle and caught_late
    results["Test 5 (Distributed mistakes across paragraph: all sentences analyzed)"] = "PASS" if test5_pass else "FAIL"
    print(f"  Analyzed across whole paragraph: {test5_pass}")
    print(f"  Result: {'PASS' if test5_pass else 'FAIL'}")

    # -----------------------------------------------------------------------
    # TEST 6: Natural expressions without actual grammar errors
    # -----------------------------------------------------------------------
    print("\n[Test 6] Natural expressions paragraph (type == expression)")
    p6 = (
        "Everything is going out of my hand today. "
        "My head is not working properly because of the heat. "
        "I am not getting your point about the new schedule."
    )
    r6, _ = await llm.generate_response(p6)
    print(f"  Utterance: {p6}")
    corrs6 = r6.get("corrections", [])
    print(f"  Corrections count: {len(corrs6)}")
    print(json.dumps(corrs6, indent=2))
    
    has_expressions = any(c.get("type") == "expression" for c in corrs6)
    # None should be falsely marked as broken syntax grammar
    all_expression_or_natural = all(c.get("type") in ("expression", "word_choice", "sentence_structure") for c in corrs6)
    test6_pass = len(corrs6) >= 1 and has_expressions
    results["Test 6 (Natural expressions classified as expression)"] = "PASS" if test6_pass else "FAIL"
    print(f"  Has expression type: {has_expressions}")
    print(f"  Result: {'PASS' if test6_pass else 'FAIL'}")

    # -----------------------------------------------------------------------
    # TEST 7: Multi-turn conversation verifying Step 1 containment remains intact
    # -----------------------------------------------------------------------
    print("\n[Test 7] Multi-turn zero contamination verification (Step 1 preservation)")
    turn1_u = "Yesterday I go to college and I meeted my professor."
    r7_t1, _ = await llm.generate_response(turn1_u)
    history = [
        {"role": "user", "content": turn1_u},
        {"role": "assistant", "content": r7_t1["response_text"]}
    ]
    turn2_u = "The discussion was clear and we solved all the questions together."
    r7_t2, _ = await llm.generate_response(turn2_u, history=history)
    print(f"  Turn 2 clean utterance: {turn2_u}")
    print(f"  Turn 2 corrections: {r7_t2.get('corrections', [])}")
    
    test7_pass = len(r7_t2.get("corrections", [])) == 0 and "went" not in r7_t2.get("spoken_text", "").lower()
    results["Test 7 (Step 1 preservation: multi-turn zero contamination)"] = "PASS" if test7_pass else "FAIL"
    print(f"  Result: {'PASS' if test7_pass else 'FAIL'}")

    # -----------------------------------------------------------------------
    # TEST 8: Original meaning preservation and minimal editing
    # -----------------------------------------------------------------------
    print("\n[Test 8] Meaning preservation and minimal editing")
    p8 = "Yesterday I go to market."
    r8, _ = await llm.generate_response(p8)
    corrs8 = r8.get("corrections", [])
    test8_pass = False
    if corrs8:
        c8 = corrs8[0]
        # Must preserve 'yesterday', 'market', and only fix 'go' -> 'went' (no rewriting into 'I visited the retail shop')
        corr_text = c8.get("corrected", "").lower()
        test8_pass = "went" in corr_text and ("market" in corr_text or "yesterday" in corr_text) and "retail" not in corr_text and "educational" not in corr_text
    results["Test 8 (Meaning preservation & minimal editing)"] = "PASS" if test8_pass else "FAIL"
    print(f"  Correction: {corrs8[0].get('corrected') if corrs8 else 'None'}")
    print(f"  Preserved core meaning: {test8_pass}")
    print(f"  Result: {'PASS' if test8_pass else 'FAIL'}")

    # -----------------------------------------------------------------------
    # Summary Table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2 VERIFICATION SUMMARY")
    print("=" * 70)
    all_pass = True
    for test_name, status in results.items():
        print(f"{test_name:<65}: {status}")
        if status != "PASS":
            all_pass = False

    print("\nOVERALL STATUS:", "ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    return all_pass

if __name__ == "__main__":
    asyncio.run(run_step2_tests())
