"""
Alex Adaptive Speech Timing and Pause Management Service.

Implements the dynamic prosody and pause-management model:
    P = B + (C * K) + (M * D)

Where:
    P = Calculated pause in milliseconds
    B = Baseline pause for the specific timing category
    C = Correction complexity (1: Simple, 2: Moderate, 3: Complex)
    K = Complexity weight (default 100 ms)
    M = Number of corrections already presented in the current turn (0, 1, 2, ...)
    D = Additional spacing per correction (default 100 ms)

Includes deterministic complexity classification, timing categories,
and safety clamps (min/max limits) so speech never feels unnaturally delayed.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
import re


class PauseCategory(str, Enum):
    NORMAL_PHRASE = "normal_phrase"                 # 150–250 ms
    SENTENCE_ENDING = "sentence_ending"             # 300–450 ms
    BEFORE_CORRECTION = "before_correction"         # 350–500 ms
    AFTER_CORRECTION_PHRASE = "after_phrase"        # 400–600 ms (between erroneous and corrected phrase)
    BETWEEN_CORRECTIONS = "between_corrections"     # 700–1000 ms (between separate correction items)
    TRANSITION_TO_REPLY = "transition_to_reply"     # 600–900 ms (after last correction before reply)
    BEFORE_EXPLANATION = "before_explanation"       # 500–800 ms
    AFTER_EXPLANATION = "after_explanation"         # 600–900 ms


# Configurable baseline defaults (in milliseconds)
BASELINE_PAUSES: Dict[PauseCategory, int] = {
    PauseCategory.NORMAL_PHRASE: 200,
    PauseCategory.SENTENCE_ENDING: 350,
    PauseCategory.BEFORE_CORRECTION: 400,
    PauseCategory.AFTER_CORRECTION_PHRASE: 450,
    PauseCategory.BETWEEN_CORRECTIONS: 750,
    PauseCategory.TRANSITION_TO_REPLY: 700,
    PauseCategory.BEFORE_EXPLANATION: 600,
    PauseCategory.AFTER_EXPLANATION: 750,
}

# Clamps to prevent awkwardly long or imperceptibly short pauses
MIN_PAUSE_MS = 150
MAX_PAUSE_MS = 1200


def calculate_adaptive_pause(
    category: PauseCategory,
    complexity: int = 1,
    correction_index: int = 0,
    k_weight_ms: int = 100,
    d_spacing_ms: int = 100,
    custom_baseline: Optional[int] = None
) -> int:
    """
    Calculates adaptive pause duration in milliseconds using:
        P = B + (C * K) + (M * D)
    
    Args:
        category: The speech timing category (e.g. BETWEEN_CORRECTIONS, AFTER_CORRECTION_PHRASE)
        complexity: Level 1 (Simple), Level 2 (Moderate), or Level 3 (Complex)
        correction_index: M = 0-based index of correction already presented
        k_weight_ms: Weight per complexity level (default 100ms)
        d_spacing_ms: Incremental spacing per correction (default 100ms)
        custom_baseline: Optional override for baseline B
    
    Returns:
        P clamped between MIN_PAUSE_MS and MAX_PAUSE_MS.
    """
    # 1. Determine baseline B
    b = custom_baseline if custom_baseline is not None else BASELINE_PAUSES.get(category, 400)
    
    # 2. Bound complexity C between 1 and 3
    c = max(1, min(3, int(complexity or 1)))
    
    # 3. Bound correction count M >= 0
    m = max(0, int(correction_index or 0))
    
    # 4. For non-correction categories (normal conversational speech), C and M don't scale indefinitely
    if category in (PauseCategory.NORMAL_PHRASE, PauseCategory.SENTENCE_ENDING):
        calculated = b
    elif category == PauseCategory.BEFORE_CORRECTION:
        # Before the first correction: subtle transition
        calculated = b + (c * (k_weight_ms // 2))
    elif category == PauseCategory.AFTER_CORRECTION_PHRASE:
        # Between "I go..." and "I WENT...": gives learner moment to contrast phrases
        calculated = b + (c * k_weight_ms)
    elif category == PauseCategory.BETWEEN_CORRECTIONS:
        # Between separate correction items: scales with both complexity and previous corrections count
        calculated = b + (c * k_weight_ms) + (m * d_spacing_ms)
    elif category == PauseCategory.TRANSITION_TO_REPLY:
        # Transition back to conversation after all corrections
        calculated = b + min(300, m * (d_spacing_ms // 2))
    else:
        calculated = b + (c * k_weight_ms) + (m * d_spacing_ms)

    # 5. Apply safety clamps
    return max(MIN_PAUSE_MS, min(MAX_PAUSE_MS, calculated))


def classify_complexity(
    original: str = "",
    corrected: str = "",
    corr_type: str = "grammar",
    explanation: str = ""
) -> int:
    """
    Classifies correction into one of three complexity levels:
        Level 1 — Simple: Verb tense, singular/plural agreement, articles, basic prepositions, simple word order
        Level 2 — Moderate: Auxiliary verbs, multiple connected errors, sentence structure, compound agreement, expressions
        Level 3 — Complex: Conditional structures, complex clauses, meaning-sensitive restructuring, multiple interacting errors
    """
    orig_clean = str(original or "").strip().lower()
    corr_clean = str(corrected or "").strip().lower()
    exp_clean = str(explanation or "").strip().lower()
    type_clean = str(corr_type or "").strip().lower()

    combined_text = f"{orig_clean} {corr_clean} {exp_clean}"

    # --- LEVEL 3: Complex ---
    level3_keywords = [
        "conditional", "if ", "would have", "could have", "should have",
        "clause", "subjunctive", "restructure", "complex sentence",
        "relative pronoun", "whom", "passive voice", "inversion"
    ]
    if any(kw in combined_text for kw in level3_keywords):
        return 3

    # Multi-word sentence restructuring where word count changes by > 3 words
    orig_words = orig_clean.split()
    corr_words = corr_clean.split()
    if len(orig_words) >= 6 and abs(len(corr_words) - len(orig_words)) >= 3:
        return 3

    # --- LEVEL 2: Moderate ---
    level2_keywords = [
        "auxiliary", "don't", "doesn't", "didn't", "haven't", "hasn't",
        "have been", "has been", "is having", "structure", "word order",
        "sentence structure", "expression", "idiom", "phrasing", "preposition"
    ]
    if type_clean in ("expression", "sentence_structure") or any(kw in combined_text for kw in level2_keywords):
        return 2

    if len(orig_words) >= 4 or len(corr_words) >= 4:
        return 2

    # --- LEVEL 1: Simple (Default) ---
    # Basic verb tense (went/go), agreement (was/were), articles (a/an/the), simple prepositions
    return 1


def build_timing_schedule(corrections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Builds the complete adaptive timing schedule for all correction segments in a turn.
    Returns a list of timing descriptor dicts.
    """
    schedule = []
    if not corrections:
        return schedule

    for idx, c in enumerate(corrections):
        comp = c.get("complexity") or classify_complexity(
            c.get("original", ""),
            c.get("corrected", ""),
            c.get("type", "grammar"),
            c.get("explanation", "")
        )
        c["complexity"] = comp

        # 1. Pause between erroneous phrase and corrected phrase
        pause_after_phrase = calculate_adaptive_pause(
            PauseCategory.AFTER_CORRECTION_PHRASE,
            complexity=comp,
            correction_index=idx
        )

        # 2. Pause before next correction or transition to reply
        is_last = (idx == len(corrections) - 1)
        if is_last:
            pause_after_corr = calculate_adaptive_pause(
                PauseCategory.TRANSITION_TO_REPLY,
                complexity=comp,
                correction_index=idx
            )
        else:
            pause_after_corr = calculate_adaptive_pause(
                PauseCategory.BETWEEN_CORRECTIONS,
                complexity=comp,
                correction_index=idx
            )

        schedule.append({
            "correction_index": idx,
            "complexity": comp,
            "pause_between_phrases_ms": pause_after_phrase,
            "pause_after_correction_ms": pause_after_corr,
            "is_last_correction": is_last
        })

    return schedule
