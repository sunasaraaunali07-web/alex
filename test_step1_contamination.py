import asyncio
import json
from app.services.llm_service import (
    LLMService,
    _phrase_exists_in_user_text,
    _validate_response,
    _parse_llm_json
)

async def run_step1_tests():
    print("=" * 65)
    print("ALEX PHASE 2 — STEP 1: PREVIOUS-TURN CONTAMINATION VERIFICATION")
    print("=" * 65)
    
    llm = LLMService()
    results = {}

    # -----------------------------------------------------------------------
    # TEST 1: Containment Unit Test — Absent phrase rejection
    # -----------------------------------------------------------------------
    print("\n[Test 1] Backend validation: Absent phrase rejection")
    mock_payload = {
        "corrections": [
            {
                "original": "Yesterday I go to market",
                "corrected": "Yesterday I went to the market",
                "explanation": "Past tense needed.",
                "type": "grammar"
            }
        ],
        "response_text": "I enjoy having a morning coffee too! What is your favorite kind?",
        "spoken_text": "You can say 'Yesterday I went to the market.' I enjoy having a morning coffee too! What is your favorite kind?"
    }
    user_msg_clean = "I am enjoying a hot cup of black coffee this morning."
    validated = _validate_response(mock_payload, user_text=user_msg_clean)
    
    test1_pass = (
        len(validated["corrections"]) == 0 and 
        validated["spoken_text"] == validated["response_text"]
    )
    results["Test 1 (Absent phrase rejection in _validate_response)"] = "PASS" if test1_pass else "FAIL"
    print(f"  Corrections count: {len(validated['corrections'])} (Expected 0)")
    print(f"  Spoken text equals response text: {validated['spoken_text'] == validated['response_text']}")
    print(f"  Result: {'PASS' if test1_pass else 'FAIL'}")

    # -----------------------------------------------------------------------
    # TEST 2: Whitespace and boundary punctuation tolerance
    # -----------------------------------------------------------------------
    print("\n[Test 2] Phrase matching tolerance: whitespace, case & punctuation")
    u_text = "Yesterday I go to market, and buy some clothes."
    t2_a = _phrase_exists_in_user_text("Yesterday   I   go", u_text)       # extra whitespace
    t2_b = _phrase_exists_in_user_text("yesterday i go", u_text)           # case insensitivity
    t2_c = _phrase_exists_in_user_text("Yesterday I go.", u_text)          # trailing punctuation added by LLM
    t2_d = _phrase_exists_in_user_text('"Yesterday I go"', u_text)        # quotes added by LLM
    t2_e = not _phrase_exists_in_user_text("Yesterday I went", u_text)      # completely absent word
    t2_f = not _phrase_exists_in_user_text("He don't know", u_text)        # absent sentence from previous turn
    
    test2_pass = t2_a and t2_b and t2_c and t2_d and t2_e and t2_f
    results["Test 2 (Whitespace, case & punctuation tolerance)"] = "PASS" if test2_pass else "FAIL"
    print(f"  Whitespace tolerant: {t2_a}")
    print(f"  Case insensitive: {t2_b}")
    print(f"  Punctuation tolerant: {t2_c} & {t2_d}")
    print(f"  Rejects absent phrases: {t2_e} & {t2_f}")
    print(f"  Result: {'PASS' if test2_pass else 'FAIL'}")

    # -----------------------------------------------------------------------
    # TEST 3: Live Turn 1 Mistake -> Turn 2 Clean Sentence (NO CONTAMINATION)
    # -----------------------------------------------------------------------
    print("\n[Test 3] Live API: Turn 1 mistake -> Turn 2 clean sentence")
    turn1_user = "Yesterday I go to college and met my professor."
    r1, _ = await llm.generate_response(turn1_user)
    print(f"  Turn 1 User: '{turn1_user}'")
    print(f"  Turn 1 Corrections: {json.dumps(r1.get('corrections'), indent=4)}")
    
    # Establish multi-turn history
    history = [
        {"role": "user", "content": turn1_user},
        {"role": "assistant", "content": r1["response_text"]}
    ]
    
    # Turn 2: Perfectly clean sentence
    turn2_user = "The meeting was very helpful and I learned a lot about our project."
    r2, _ = await llm.generate_response(turn2_user, history=history)
    print(f"  Turn 2 User: '{turn2_user}'")
    print(f"  Turn 2 Corrections: {json.dumps(r2.get('corrections'), indent=4)}")
    print(f"  Turn 2 Spoken Text: '{r2.get('spoken_text')}'")
    
    corrs_t2 = r2.get("corrections", [])
    repeated_t1 = any("go" in str(c).lower() or "went" in str(c).lower() for c in corrs_t2)
    repeated_spoken = "went" in r2.get("spoken_text", "").lower() and "college" in r2.get("spoken_text", "").lower()
    test3_pass = len(corrs_t2) == 0 and not repeated_t1 and not repeated_spoken
    results["Test 3 (Turn 1 mistake -> Turn 2 clean sentence: Zero contamination)"] = "PASS" if test3_pass else "FAIL"
    print(f"  Result: {'PASS' if test3_pass else 'FAIL'}")

    # -----------------------------------------------------------------------
    # TEST 4: Live Turn 1 Mistake -> Turn 2 New Mistake (ONLY Turn 2 corrected)
    # -----------------------------------------------------------------------
    print("\n[Test 4] Live API: Turn 1 mistake -> Turn 2 new mistake")
    turn2_mistake_user = "My project partner is nice, but he don't understand the backend logic."
    r4, _ = await llm.generate_response(turn2_mistake_user, history=history)
    print(f"  Turn 2 User (New Mistake): '{turn2_mistake_user}'")
    print(f"  Turn 2 Corrections: {json.dumps(r4.get('corrections'), indent=4)}")
    print(f"  Turn 2 Spoken Text: '{r4.get('spoken_text')}'")
    
    corrs_t4 = r4.get("corrections", [])
    has_doesnt = any("doesn't" in c.get("corrected", "").lower() or "does not" in c.get("corrected", "").lower() for c in corrs_t4)
    # Crucial check: must NOT re-correct turn 1's "go/went"
    has_turn1_leak = any("go" in c.get("original", "").lower() or "college" in c.get("original", "").lower() for c in corrs_t4)
    test4_pass = len(corrs_t4) >= 1 and has_doesnt and not has_turn1_leak
    results["Test 4 (Turn 1 mistake -> Turn 2 new mistake: Only Turn 2 corrected)"] = "PASS" if test4_pass else "FAIL"
    print(f"  Corrected Turn 2 ('he don't' -> 'he doesn't'): {has_doesnt}")
    print(f"  No leakage from Turn 1 ('go'/'college'): {not has_turn1_leak}")
    print(f"  Result: {'PASS' if test4_pass else 'FAIL'}")

    # -----------------------------------------------------------------------
    # TEST 5: Existing valid corrections continue working
    # -----------------------------------------------------------------------
    print("\n[Test 5] Existing valid corrections regression check")
    test_phrase = "I still not completed mine."
    r5, _ = await llm.generate_response(test_phrase)
    corrs_t5 = r5.get("corrections", [])
    has_havent = any("haven't" in c.get("corrected", "").lower() or "have not" in c.get("corrected", "").lower() for c in corrs_t5)
    test5_pass = len(corrs_t5) >= 1 and has_havent
    results["Test 5 (Existing valid corrections regression check)"] = "PASS" if test5_pass else "FAIL"
    print(f"  Corrected 'still not completed' -> 'haven't completed': {has_havent}")
    print(f"  Result: {'PASS' if test5_pass else 'FAIL'}")

    # -----------------------------------------------------------------------
    # Summary Table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 1 VERIFICATION SUMMARY")
    print("=" * 65)
    all_pass = True
    for test_name, status in results.items():
        print(f"{test_name:<60}: {status}")
        if status != "PASS":
            all_pass = False

    print("\nOVERALL STATUS:", "ALL TESTS PASSED" if all_pass else "TESTS FAILED")
    return all_pass

if __name__ == "__main__":
    asyncio.run(run_step1_tests())
