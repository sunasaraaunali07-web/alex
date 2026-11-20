import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.pause_manager import (
    calculate_adaptive_pause,
    classify_complexity,
    build_timing_schedule,
    PauseCategory,
    MIN_PAUSE_MS,
    MAX_PAUSE_MS
)

def test_pause_manager():
    # 1. Test user's exact example:
    # B = 400ms, C = 2, K = 100ms, M = 1, D = 100ms -> P = 700ms
    p1 = calculate_adaptive_pause(
        PauseCategory.BETWEEN_CORRECTIONS,
        complexity=2,
        correction_index=1,
        custom_baseline=400,
        k_weight_ms=100,
        d_spacing_ms=100
    )
    print(f"User Example Calculation P: {p1}ms (Expected: 700ms)")
    assert p1 == 700, f"Expected 700, got {p1}"

    # 2. Test bounds clamping
    p_min = calculate_adaptive_pause(PauseCategory.NORMAL_PHRASE, custom_baseline=50)
    assert p_min >= MIN_PAUSE_MS, f"Should clamp to at least {MIN_PAUSE_MS}, got {p_min}"
    p_max = calculate_adaptive_pause(PauseCategory.BETWEEN_CORRECTIONS, complexity=3, correction_index=10, custom_baseline=1500)
    assert p_max <= MAX_PAUSE_MS, f"Should clamp to at most {MAX_PAUSE_MS}, got {p_max}"
    print("PASS: Bounds clamping verified")

    # 3. Test complexity classification
    c_simple = classify_complexity("Yesterday I go", "Yesterday I went", "grammar", "past tense")
    assert c_simple == 1, f"Expected Level 1 for simple tense, got {c_simple}"

    c_moderate = classify_complexity("He don't know", "He doesn't know", "grammar", "auxiliary verb")
    assert c_moderate == 2, f"Expected Level 2 for auxiliary, got {c_moderate}"

    c_complex = classify_complexity("If I was knowing, I would have went", "If I had known, I would have gone", "grammar", "conditional clause")
    assert c_complex == 3, f"Expected Level 3 for conditional clause, got {c_complex}"
    print("PASS: Complexity classification verified (Levels 1, 2, 3)")

    # 4. Test schedule builder
    sample_corrs = [
        {"original": "I go", "corrected": "I went", "type": "grammar"},
        {"original": "He don't", "corrected": "He doesn't", "type": "grammar"},
        {"original": "If I was there", "corrected": "If I had been there", "type": "grammar", "explanation": "conditional"}
    ]
    sched = build_timing_schedule(sample_corrs)
    assert len(sched) == 3
    # Check that spacing increases for later corrections (M=0 vs M=1)
    p_c0 = sched[0]["pause_after_correction_ms"]
    p_c1 = sched[1]["pause_after_correction_ms"]
    print(f"Pause after Corr 0: {p_c0}ms, after Corr 1: {p_c1}ms")
    assert p_c1 > p_c0, f"Expected increasing pause for second correction, got {p_c1} vs {p_c0}"
    assert sched[2]["is_last_correction"] is True
    print("PASS: Timing schedule builder verified")

    print("\nALL PAUSE MANAGER TESTS PASSED!")

if __name__ == "__main__":
    test_pause_manager()
