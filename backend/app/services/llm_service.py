import asyncio
import difflib
import json
import re
import time
from google import genai
from groq import AsyncGroq
from app.core.config import get_settings
from app.services.pause_manager import classify_complexity

# ---------------------------------------------------------------------------
# System Prompt — Alex as grammar-aware conversational friend
# ---------------------------------------------------------------------------
FRIEND_SYSTEM_PROMPT = """You are Alex, a warm, intelligent and encouraging English-speaking friend helping the user practise conversational English.

## 1. STRICT IDENTITY & BEHAVIORAL BOUNDARIES
- You are Alex, an English-speaking conversational practice companion.
- STRICT PROHIBITION: You are NEVER an AI coding assistant, HTML helper, software engineer, or programming instructor.
- If the user mentions software, websites, buttons, colors, code, or prompts, or gives feedback about practicing with you, NEVER provide HTML/CSS code, programming syntax, or technical tutorials! Always answer warmly, naturally, and supportively as a conversation partner in simple spoken English.
- Do not claim human experiences, a physical body, or biographical facts.
- If asked who made you: say you are Alex, created to help practise conversational English.

## 2. YOUR DUAL ROLE (MANDATORY CORRECTION-FIRST INSTRUCTION)
1. CRITICAL PRIORITY: On EVERY user turn, you MUST thoroughly audit the user's latest message for all genuine English mistakes (tense, subject-verb agreement, prepositions, articles, auxiliaries, question word order, missing words).
   - If genuine grammatical mistakes exist, output at most 2 to 3 of the most impactful errors in "corrections".
   - Split multiple mistakes into separate short cards.
2. In "response_text", provide your natural, intelligent, and calibrated conversational answer continuing the chat (1 to 3 natural spoken sentences).
   - STRICT RULE: NEVER mention, recap, or lecture about grammar mistakes or rules in "response_text"! All grammar coaching belongs EXCLUSIVELY in "corrections".

## 3. CORRECTION CARD RULES (COMPACT 2 TO 5 WORDS MAXIMUM)
- "original": ONLY the specific erroneous phrase with 1-2 words of natural context (2 to 5 words MAXIMUM, e.g. 'what I trying', 'still not able', 'he don't understand', 'start arguing with'). NEVER dump the entire surrounding sentence!
- "corrected": ONLY the replacement phrase with 1-2 words of natural context (2 to 5 words MAXIMUM, e.g. 'what I am trying', 'am still not able', 'he doesn't understand', 'starts arguing'). NEVER the entire sentence!
- "spoken_comparison": Strictly concise spoken contrast:
  * First card: "In that sentence, say '{corrected}' instead of '{original}'."
  * Second card: "Also, say '{corrected}' instead of '{original}'."
  * Third card: "And say '{corrected}' rather than '{original}'."
  * NEVER include 'because' or grammar explanations in "spoken_comparison"!
- "emphasized_word": The key changed word in uppercase (e.g. 'DOESN'T', 'WENT').
- "explanation": One brief, friendly sentence explaining why (for the written card ONLY, e.g. "Use 'doesn't' with third-person singular subjects like 'he'.").
- "type": "grammar" (for all actual grammatical errors) or "expression" (for awkward but valid phrasing).
- If the latest user message is grammatically correct, "corrections" MUST be [].
- Never replace acceptable words with synonyms.
- Meta-language: If user cites example phrases or gives feedback, recognize they are discussing examples and do NOT treat quoted examples as mistakes.

## 4. RESPONSE FORMAT
Respond with ONLY a valid JSON object. No markdown fences, no extra text outside the JSON:
{
  "response_text": "your purely conversational reply to the user answering their question or continuing the chat without any grammar mention",
  "corrections": [
    {
      "original": "he don't understand",
      "corrected": "he doesn't understand",
      "spoken_comparison": "In that sentence, say 'he doesn't understand' instead of 'he don't understand'.",
      "emphasized_word": "DOESN'T",
      "explanation": "Use 'doesn't' with third-person singular subjects like 'he'.",
      "type": "grammar",
      "complexity": 1
    }
  ]
}"""


# ---------------------------------------------------------------------------
# Phrase containment and matching helpers (Prevents Previous-Turn Contamination)
# ---------------------------------------------------------------------------

def _phrase_exists_in_user_text(phrase: str, user_text: str) -> bool:
    """
    Verifies that `phrase` exists within `user_text`.
    Case-insensitive, tolerant of whitespace variations, boundary punctuation,
    and internal punctuation differences.
    Guarantees that corrections from earlier turns or hallucinations cannot leak in.
    """
    if not phrase or not user_text:
        return False

    # Normalize whitespace and lowercase
    p_norm = re.sub(r"\s+", " ", phrase).strip().lower()
    u_norm = re.sub(r"\s+", " ", user_text).strip().lower()

    # 1. Direct normalized substring match
    if p_norm in u_norm:
        return True

    # 2. Punctuation-trimmed match (e.g. LLM added trailing period or boundary quotes)
    p_stripped = p_norm.strip(".,!?;:\"'()[]{}")
    if p_stripped and p_stripped in u_norm:
        return True

    # 3. Word-sequence match ignoring internal punctuation
    p_clean = re.sub(r"[^\w\s]", "", p_norm)
    u_clean = re.sub(r"[^\w\s]", "", u_norm)
    p_words = p_clean.split()
    u_words = u_clean.split()

    if p_words and u_words and len(p_words) <= len(u_words):
        len_p = len(p_words)
        for i in range(len(u_words) - len_p + 1):
            if u_words[i:i + len_p] == p_words:
                return True

    return False


def _overlaps_uncertain_segment(phrase: str, uncertain_segments: list[str]) -> bool:
    """
    Checks whether `phrase` is contained in or overlaps with any uncertain ASR segment.
    Protects user from being penalized for speech recognition errors.
    """
    if not phrase or not uncertain_segments:
        return False

    p_norm = re.sub(r"\s+", " ", phrase).strip().lower()
    p_words = re.findall(r"\b\w+\b", p_norm)
    if not p_words:
        return False

    for u_seg in uncertain_segments:
        if not u_seg:
            continue
        u_norm = re.sub(r"\s+", " ", u_seg).strip().lower()

        # 1. Direct containment in either direction
        if _phrase_exists_in_user_text(phrase, u_seg) or _phrase_exists_in_user_text(u_seg, phrase):
            return True

        # 2. Word-level check: if single-word phrase is inside uncertain segment
        u_words = set(re.findall(r"\b\w+\b", u_norm))
        if len(p_words) == 1 and p_words[0] in u_words:
            return True

        # 3. N-gram check: if any 2-word phrase from candidate correction is inside uncertain segment
        if len(p_words) >= 2:
            for i in range(len(p_words) - 1):
                bigram = f"{p_words[i]} {p_words[i+1]}"
                if bigram in u_norm:
                    return True

    return False


# ---------------------------------------------------------------------------
# JSON parsing and validation helpers
# ---------------------------------------------------------------------------

def _clean_token(w: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", w).lower()


def _extract_phrase_from_comparison(spoken_comp: str, user_text: str) -> tuple[str, str] | None:
    """
    Extracts the minimal target phrase from explicit spoken comparison like:
    - say 'what I am trying' instead of 'what I trying'
    - say 'went' rather than 'go'
    - use 'two apples' instead of 'two apple'
    Returns (original, corrected) if original exists in user_text and is concise.
    """
    if not spoken_comp:
        return None
    # Pattern 1: say/use 'X' instead of/rather than/not 'Y'
    m = re.search(r"(?:say|use)\s+['\"]([^'\"]+)['\"]\s+(?:instead of|rather than|not)\s+['\"]([^'\"]+)['\"]", spoken_comp, re.IGNORECASE)
    if m:
        corr = m.group(1).strip()
        orig = m.group(2).strip()
        if orig and _phrase_exists_in_user_text(orig, user_text) and len(orig.split()) <= 6:
            return orig, corr

    # Pattern 2: instead of/rather than 'Y', say/use 'X'
    m2 = re.search(r"(?:instead of|rather than)\s+['\"]([^'\"]+)['\"].*?(?:say|use)\s+['\"]([^'\"]+)['\"]", spoken_comp, re.IGNORECASE)
    if m2:
        orig = m2.group(1).strip()
        corr = m2.group(2).strip()
        if orig and _phrase_exists_in_user_text(orig, user_text) and len(orig.split()) <= 6:
            return orig, corr

    return None


def _isolate_diff_cluster(original: str, corrected: str, cluster_idx: int = 0) -> tuple[str, str]:
    """
    If original and corrected are full sentences (> 5 words), aligns them by tokens
    and isolates only the specific changed difference cluster with 1 word of surrounding context.
    Prevents whole sentences from leaking into correction cards.
    """
    orig_words = original.strip().split()
    corr_words = corrected.strip().split()

    if len(orig_words) <= 5 or len(corr_words) <= 5:
        return original.strip(), corrected.strip()

    matcher = difflib.SequenceMatcher(None, [_clean_token(w) for w in orig_words], [_clean_token(w) for w in corr_words])
    opcodes = matcher.get_opcodes()

    diff_clusters = []
    current_cluster = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag != 'equal':
            current_cluster.append((i1, i2, j1, j2))
        else:
            if current_cluster:
                if (i2 - i1) >= 2:
                    diff_clusters.append(current_cluster)
                    current_cluster = []
                else:
                    current_cluster.append((i1, i2, j1, j2))
    if current_cluster:
        diff_clusters.append(current_cluster)

    if not diff_clusters:
        return original.strip(), corrected.strip()

    target = diff_clusters[cluster_idx] if cluster_idx < len(diff_clusters) else diff_clusters[-1]
    min_i = min(c[0] for c in target)
    max_i = max(c[1] for c in target)
    min_j = min(c[2] for c in target)
    max_j = max(c[3] for c in target)

    start_orig = max(0, min_i - 1)
    end_orig = min(len(orig_words), max_i + 1)
    start_corr = max(0, min_j - 1)
    end_corr = min(len(corr_words), max_j + 1)

    isolated_orig = " ".join(orig_words[start_orig:end_orig]).strip()
    isolated_corr = " ".join(corr_words[start_corr:end_corr]).strip()

    if isolated_orig and isolated_corr:
        return isolated_orig, isolated_corr

    return original.strip(), corrected.strip()


def _clean_spoken_comparison(spoken_comp: str, orig: str, corr: str, idx: int) -> str:
    """
    Ensures spoken_comparison is strictly a concise comparison sentence.
    Always guarantees both the corrected and original phrases are articulated.
    Never truncates phrases containing words like 'because', 'since', etc.
    """
    clean_orig = orig.strip().strip("'\"")
    clean_corr = corr.strip().strip("'\"")

    # If spoken_comp already has a clean, complete comparison containing both orig and corr, keep it
    if spoken_comp and clean_orig.lower() in spoken_comp.lower() and clean_corr.lower() in spoken_comp.lower():
        # Only strip trailing explanation if it is OUTSIDE the quotes at the end of the sentence
        clean = re.sub(r"['\"]\s*,\s*(?:because|since|as you need|to form|due to)\b.*$", "'", spoken_comp, flags=re.IGNORECASE).strip()
        clean = clean.rstrip(".,;:")
        if any(m in clean.lower() for m in ["instead of", "rather than", "not '", "not \""]) and len(clean.split()) <= 15:
            return f"{clean}."

    # Always generate guaranteed, complete spoken comparison
    if idx == 0:
        return f"In that sentence, say '{clean_corr}' instead of '{clean_orig}'."
    elif idx == 1:
        return f"Also, say '{clean_corr}' instead of '{clean_orig}'."
    else:
        return f"And say '{clean_corr}' rather than '{clean_orig}'."


def _trim_common_sentence_context(original: str, corrected: str) -> tuple[str, str]:
    """
    If the LLM provided an entire surrounding sentence in original/corrected,
    trims large common leading and trailing words so the card focuses on the
    meaningful error phrase without chopping internal grammatical structure.
    """
    orig_words = original.strip().split()
    corr_words = corrected.strip().split()

    if len(orig_words) <= 5 or len(corr_words) <= 5:
        return original.strip(), corrected.strip()

    # Find common prefix words
    p = 0
    while p < len(orig_words) and p < len(corr_words) and _clean_token(orig_words[p]) == _clean_token(corr_words[p]):
        p += 1

    # Find common suffix words
    s = 0
    while s < (len(orig_words) - p) and s < (len(corr_words) - p) and _clean_token(orig_words[len(orig_words) - 1 - s]) == _clean_token(corr_words[len(corr_words) - 1 - s]):
        s += 1

    # Only trim if there are substantial common context words (>= 3 words)
    if p + s >= 3:
        trimmed_orig = " ".join(orig_words[p:len(orig_words) - s]).strip()
        trimmed_corr = " ".join(corr_words[p:len(corr_words) - s]).strip()
        if trimmed_orig and trimmed_corr:
            return trimmed_orig, trimmed_corr

    return original.strip(), corrected.strip()


def extract_truncated_response_text(raw: str) -> str:
    """
    Extracts response_text or conversational content from truncated or malformed JSON.
    Recovers valid answers when the model hits max token limits before closing JSON braces.
    """
    if not raw:
        return ""
    # 1. Direct JSON load attempt if partially valid
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            for k in ["response_text", "response", "answer", "reply", "message", "content", "text"]:
                if data.get(k) and isinstance(data[k], str) and data[k].strip():
                    return data[k].strip()
    except Exception:
        pass

    # 2. Extract string content via regex
    for key in ["response_text", "response", "answer", "reply", "message", "content", "text"]:
        pattern = rf'"{key}"\s*:\s*"'
        match = re.search(pattern, raw)
        if match:
            start_pos = match.end()
            extracted = raw[start_pos:]
            # Find closing unescaped quote
            end_match = re.search(r'(?<!\\)"', extracted)
            if end_match:
                extracted = extracted[:end_match.start()]
            else:
                extracted = re.sub(r'\\+$', '', extracted).strip()
            try:
                unescaped = json.loads(f'"{extracted}"')
                if unescaped.strip():
                    return unescaped.strip()
            except Exception:
                extracted = extracted.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                if extracted.strip():
                    return extracted.strip()
    return ""


def _parse_llm_json(raw: str, user_text: str = "", asr_metadata: dict = None) -> dict:
    """
    Robustly parse structured JSON from LLM output.
    Handles: clean JSON, JSON wrapped in markdown fences, stray text before/after.
    Recovers truncated/unclosed JSON content before falling back.
    """
    if not raw:
        return _fallback_response("", [])

    text = raw.strip()

    # Strip markdown code fences if present (``` or ```json)
    if "```" in text:
        lines = text.split("\n")
        cleaned = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(cleaned).strip()

    # Attempt 1: direct parse
    try:
        data = json.loads(text)
        return _validate_response(data, user_text, asr_metadata)
    except json.JSONDecodeError:
        pass

    # Attempt 2: find the outermost JSON object in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if 0 <= start < end:
        try:
            data = json.loads(text[start:end])
            return _validate_response(data, user_text, asr_metadata)
        except json.JSONDecodeError:
            pass

    # Attempt 3: Salvage response_text from truncated / unclosed JSON
    extracted = extract_truncated_response_text(text)
    if extracted:
        return _validate_response({"response_text": extracted, "corrections": []}, user_text, asr_metadata)

    # Complete fallback: treat raw text as plain conversational response
    return _fallback_response(text, [])


def _validate_response(data: dict, user_text: str = "", asr_metadata: dict = None) -> dict:
    """
    Ensure required fields exist, have correct types, and corrections are meaningful.
    Enforces that every correction's 'original' phrase actually exists in the latest user_text.
    Rejects any correction whose original phrase is not from the current message.
    Avoids duplicate or redundant overlapping corrections.
    Protects against ASR uncertainty: suppresses all corrections if turn is globally uncertain,
    and discards corrections overlapping with uncertain speech segments.
    """
    if not isinstance(data, dict):
        return _fallback_response("", [])

    asr_metadata = asr_metadata or {}

    # Check global uncertainty
    avg_lp = asr_metadata.get("avg_logprob")
    nsp = asr_metadata.get("no_speech_prob")
    is_globally_uncertain = asr_metadata.get("is_globally_uncertain", False)
    if not is_globally_uncertain:
        is_globally_uncertain = (
            (avg_lp is not None and avg_lp < -0.85) or
            (nsp is not None and nsp > 0.40)
        )

    # Robust extraction of response_text across standard and alternative keys
    response_text = ""
    for k in ["response_text", "response", "answer", "reply", "message", "content", "text"]:
        val = data.get(k)
        if val and isinstance(val, str) and val.strip():
            response_text = val.strip()
            break
        elif val and isinstance(val, (dict, list)):
            response_text = json.dumps(val, indent=2)
            break

    if not response_text:
        # Check if the LLM partitioned its answer into custom section keys
        sections = [
            f"**{k.replace('_', ' ').title()}:** {v.strip()}"
            for k, v in data.items()
            if k not in ["corrections", "spoken_text", "timing_schedule", "speech_segments"]
            and isinstance(v, str) and v.strip()
        ]
        if sections:
            response_text = "\n\n".join(sections)

    if not response_text:
        response_text = "That's interesting! Tell me more."

    spoken_text = str(data.get("spoken_text", "")).strip()
    if not spoken_text:
        spoken_text = response_text

    # If entire audio is globally uncertain, suppress all corrections deterministically
    if is_globally_uncertain:
        return {
            "corrections": [],
            "response_text": response_text,
            "spoken_text": response_text
        }

    raw_corrections = data.get("corrections", [])
    if not isinstance(raw_corrections, list):
        raw_corrections = []

    uncertain_segments = asr_metadata.get("uncertain_segments", [])
    valid_corrections = []
    seen_normalized_originals = set()

    user_text_lower = (user_text or "").lower()

    for c in raw_corrections:
        if not isinstance(c, dict):
            continue
        original = str(c.get("original", "")).strip()
        corrected = str(c.get("corrected", "")).strip()
        spoken_comp = str(c.get("spoken_comparison", "")).strip()

        # Step A: If spoken_comparison contains a clean compact comparison that exists in user_text, use it
        extracted = _extract_phrase_from_comparison(spoken_comp, user_text)
        if extracted and (len(original.split()) > 5 or not _phrase_exists_in_user_text(original, user_text)):
            original, corrected = extracted

        # Step B: If original is still a long sentence (> 5 words), trim common context or isolate diff cluster
        if len(original.split()) > 5:
            original, corrected = _trim_common_sentence_context(original, corrected)
            if len(original.split()) > 5:
                cluster_idx = len(valid_corrections)
                original, corrected = _isolate_diff_cluster(original, corrected, cluster_idx)

        # 1. Basic sanity: must be non-empty and non-identical
        if not original or not corrected or original.lower() == corrected.lower():
            continue

        # 2. Current-turn containment check:
        # If user_text is provided, original MUST exist in user_text (prevents contamination)
        if user_text and not _phrase_exists_in_user_text(original, user_text):
            continue

        # 3. ASR segment-level uncertainty protection:
        # If the phrase belongs to or overlaps an uncertain ASR segment, discard it to avoid penalizing ASR errors
        if uncertain_segments and _overlaps_uncertain_segment(original, uncertain_segments):
            continue

        # 4. Thorough deduplication and overlap prevention:
        norm_orig = re.sub(r"\s+", " ", original).strip().lower()
        norm_corr = re.sub(r"\s+", " ", corrected).strip().lower()
        if norm_orig in seen_normalized_originals:
            continue

        # Prevent duplicate or overlapping corrections on the same target mistake
        is_overlapping = False
        for existing in valid_corrections:
            ex_orig = re.sub(r"\s+", " ", existing["original"]).strip().lower()
            ex_corr = re.sub(r"\s+", " ", existing["corrected"]).strip().lower()
            if norm_orig == ex_orig:
                is_overlapping = True
                break
            # If one phrase is a substring of the other (e.g. "go" vs "go to market")
            # and that target occurs only once in the user utterance, they target the same error
            if (norm_orig in ex_orig or ex_orig in norm_orig):
                orig_target_word = norm_orig.split()[0] if norm_orig else ""
                if orig_target_word and user_text_lower.count(orig_target_word) <= 1:
                    is_overlapping = True
                    break
            # Identical correction replacement
            if norm_corr == ex_corr and (norm_orig in ex_orig or ex_orig in norm_orig):
                is_overlapping = True
                break

        if is_overlapping:
            continue

        seen_normalized_originals.add(norm_orig)

        corr_type = str(c.get("type", "grammar")).strip()
        explanation = str(c.get("explanation", "")).strip()

        # Identify changed token if emphasized_word is missing
        emph_word = str(c.get("emphasized_word", "")).strip()
        if not emph_word:
            orig_tokens = original.split()
            corr_tokens = corrected.split()
            for ct in corr_tokens:
                ct_clean = ct.strip(".,!?;:\"'")
                if ct_clean.lower() not in [ot.strip(".,!?;:\"'").lower() for ot in orig_tokens]:
                    emph_word = ct_clean.upper()
                    break
            if not emph_word and corr_tokens:
                emph_word = corr_tokens[-1].strip(".,!?;:\"'").upper()

        # Phrase-level spoken fields for backward compatibility
        spoken_orig = str(c.get("spoken_original", "")).strip()
        spoken_corr = str(c.get("spoken_corrected", "")).strip()
        if not spoken_orig:
            spoken_orig = f"{original}..."
        elif not spoken_orig.endswith("..."):
            spoken_orig = f"{spoken_orig}..."

        if not spoken_corr:
            if emph_word and emph_word.lower() in corrected.lower():
                spoken_corr = re.sub(re.escape(emph_word), emph_word.upper(), corrected, flags=re.IGNORECASE)
            else:
                spoken_corr = corrected

        # Full original and corrected sentences for written cards
        full_orig = str(c.get("full_original_sentence", "")).strip()
        full_corr = str(c.get("full_corrected_sentence", "")).strip()

        if not full_orig and user_text:
            sentences = re.split(r"(?<=[.!?])\s+", user_text)
            for s in sentences:
                if _phrase_exists_in_user_text(original, s):
                    full_orig = s.strip()
                    break
            if not full_orig:
                full_orig = user_text.strip()

        if not full_corr:
            if full_orig and original:
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                full_corr = pattern.sub(corrected, full_orig)
            else:
                full_corr = corrected

        complexity = c.get("complexity")
        if not complexity or complexity not in (1, 2, 3):
            complexity = classify_complexity(original, corrected, corr_type, explanation)

        # Spoken explicit comparison: strictly concise, NO "Why" in spoken audio!
        spoken_comp = _clean_spoken_comparison(spoken_comp, original, corrected, len(valid_corrections))

        valid_corrections.append({
            "original": original,
            "corrected": corrected,
            "spoken_original": spoken_orig,
            "spoken_corrected": spoken_corr,
            "spoken_comparison": spoken_comp,
            "emphasized_word": emph_word,
            "full_original_sentence": full_orig,
            "full_corrected_sentence": full_corr,
            "explanation": explanation,
            "type": corr_type,
            "complexity": complexity
        })

    # Unify full_corrected_sentence across corrections that share the same sentence
    # Ensures the user sees ONE complete, fully-corrected sentence with all fixes applied together
    if valid_corrections:
        # Group by full_original_sentence
        groups = {}
        for vc in valid_corrections:
            fo = vc["full_original_sentence"]
            if fo not in groups:
                groups[fo] = []
            groups[fo].append(vc)

        for fo, group_items in groups.items():
            if len(group_items) > 1:
                # Find the most comprehensive full_corrected_sentence
                best_fc = ""
                for it in group_items:
                    fc = it["full_corrected_sentence"]
                    # Check how many corrected phrases are present
                    matches = sum(1 for other in group_items if other["corrected"].lower() in fc.lower())
                    if matches == len(group_items):
                        best_fc = fc
                        break
                    elif len(fc) > len(best_fc):
                        best_fc = fc

                # If no single full_corrected_sentence has all corrections, synthesize one sequentially
                if not best_fc or not all(it["corrected"].lower() in best_fc.lower() for it in group_items):
                    synth = fo
                    for it in group_items:
                        synth = re.sub(re.escape(it["original"]), it["corrected"], synth, flags=re.IGNORECASE)
                    best_fc = synth

                # Assign the unified sentence to all items in this group
                for it in group_items:
                    it["full_corrected_sentence"] = best_fc

    # Deduplication and elimination of overlapping "More Natural Phrasing" (expression) cards
    # If a grammar correction exists for a sentence or word, DO NOT duplicate it as an expression correction!
    grammar_items = [vc for vc in valid_corrections if vc["type"] == "grammar"]
    if grammar_items:
        non_redundant_items = []
        for vc in valid_corrections:
            if vc["type"] != "grammar":
                v_orig = vc["original"].lower().strip()
                v_corr = vc["corrected"].lower().strip()
                v_fo = vc["full_original_sentence"].lower().strip()
                v_words_orig = set(re.findall(r"\b\w+\b", v_orig))
                v_words_corr = set(re.findall(r"\b\w+\b", v_corr))

                is_dup = False
                for g in grammar_items:
                    g_orig = g["original"].lower().strip()
                    g_corr = g["corrected"].lower().strip()
                    g_fo = g["full_original_sentence"].lower().strip()
                    g_words_orig = set(re.findall(r"\b\w+\b", g_orig))
                    g_words_corr = set(re.findall(r"\b\w+\b", g_corr))

                    # 1. Direct containment or word overlap in original or corrected
                    if (v_orig in g_orig or g_orig in v_orig or v_corr in g_corr or g_corr in v_corr):
                        is_dup = True
                        break
                    if (v_words_orig & g_words_orig) or (v_words_corr & g_words_corr):
                        is_dup = True
                        break
                    # 2. Same sentence target: if the sentence already has a grammar mistake,
                    # eliminate optional expression advice for the same sentence to avoid overwhelming or repeating
                    if v_fo and g_fo and v_fo == g_fo:
                        is_dup = True
                        break

                if not is_dup:
                    non_redundant_items.append(vc)
            else:
                non_redundant_items.append(vc)
        valid_corrections = non_redundant_items

    # Programmatic sanitization: strip any grammar lecturing or recapping from response_text
    grammar_leak_patterns = [
        r"^(?:also,?\s+)?(?:remember(?:,| that)?\s+)?(?:in english,?\s+)?(?:when talking about|for past actions|in the past|we use past tense|you should say|instead of saying|make sure to use|it'?s better to say|be sure to say|don't forget to use)[^.!?\n]*[.!?\n]\s*",
        r"^(?:in that sentence,?\s+)?(?:you should say|say|use) ['\"].*?['\"] instead of ['\"].*?['\"][^.!?\n]*[.!?\n]\s*",
        r"^since you(?:'re| are) talking about (?:the past|yesterday)[^.!?\n]*[.!?\n]\s*",
        r"^(?:just a (?:quick )?(?:reminder|note):?\s+)[^.!?\n]*[.!?\n]\s*",
        r"^(?:make sure to say|always remember to say)[^.!?\n]*[.!?\n]\s*"
    ]
    cleaned_resp = response_text
    for pat in grammar_leak_patterns:
        cleaned_resp = re.sub(pat, "", cleaned_resp, flags=re.IGNORECASE).strip()
    if cleaned_resp:
        response_text = cleaned_resp

    # Guardrail: Strip any accidental HTML/CSS syntax or programming tutorial advice
    if re.search(r"<\s*span|style\s*=|color\s*:\s*(?:green|blue|red)|`\s*color\s*:", response_text, re.IGNORECASE):
        clean_lines = [
            line for line in response_text.split("\n")
            if not re.search(r"style\s*=|color\s*:\s*|HTML|CSS|`color", line, re.IGNORECASE)
        ]
        cleaned_text = "\n".join(clean_lines).strip()
        response_text = cleaned_text or "I completely understand what you mean! Getting comfortable speaking English is all about keeping things clear and natural. Tell me more about what you're working on!"

    # Build natural spoken_text: spoken comparisons followed by conversational reply
    if valid_corrections:
        spoken_comparisons = [c["spoken_comparison"] for c in valid_corrections]
        spoken_text = " ... ".join(spoken_comparisons) + f" ... {response_text}"
    else:
        spoken_text = response_text

    return {
        "corrections": valid_corrections,
        "response_text": response_text,
        "spoken_text": spoken_text
    }


def _fallback_response(raw_text: str, corrections: list = None) -> dict:
    """
    Safe fallback when JSON parsing completely fails.
    Treats raw_text as a plain conversational response.
    Attempts to salvage conversational response text before defaulting.
    """
    corrections = corrections or []
    extracted = extract_truncated_response_text(raw_text)
    if extracted:
        text = extracted
    else:
        text = (raw_text or "I see! Tell me more about that.").strip()
        # Remove JSON-like artifacts that may have leaked
        if text.startswith("{") or text.startswith("["):
            text = "I see! Tell me more about that."
    # Truncate safely (allow up to 3000 chars to avoid truncating deep multi-part answers)
    text = text[:3000]
    return {
        "corrections": corrections,
        "response_text": text,
        "spoken_text": text
    }


def _determine_max_tokens(user_text: str, is_openai: bool) -> int:
    """
    Dynamically determines sensible output-token limits based on question complexity,
    intent, and explicit length instructions, while ensuring sufficient headroom for
    grammar corrections and avoiding truncation of complex answers.
    """
    text_lower = user_text.lower().strip()

    # Explicit brevity instructions or simple calculations / direct short questions
    is_explicit_short = any(p in text_lower for p in [
        "one sentence", "in short", "briefly", "shortly", "keep it short",
        "as short as possible", "quick answer", "in a nutshell", "few words"
    ])
    is_calculation = bool(re.search(r'\b(\d+\s*[\+\-\*\/\%]\s*\d+|what is \d+|calculate)\b', text_lower))
    is_simple_query = len(text_lower.split()) <= 6 and not any(k in text_lower for k in [
        "compare", "roadmap", "difference", "explain", "why", "how to"
    ])

    # Complex multi-part queries
    is_complex = any(k in text_lower for k in [
        "roadmap", "compare", "difference between", "in detail", "explain fully",
        "step by step", "pros and cons", "career", "architecture", "recommend for me and why"
    ]) or len(text_lower.split()) > 25

    if is_explicit_short or is_calculation or is_simple_query:
        return 800 if is_openai else 450
    elif is_complex:
        return 1800 if is_openai else 900
    else:
        return 1000 if is_openai else 550


# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------

class LLMService:
    def __init__(self):
        self._async_groq_client = None

    def _get_async_groq(self, api_key: str) -> AsyncGroq:
        """Lazy-initialised, cached AsyncGroq client."""
        if not self._async_groq_client:
            self._async_groq_client = AsyncGroq(api_key=api_key)
        return self._async_groq_client

    async def generate_response(
        self,
        user_text: str,
        history: list[dict] = None,
        asr_metadata: dict = None,
        user_memory_context: str = ""
    ) -> tuple[dict, int]:
        """
        Generates a structured correction + conversational response.

        Args:
            user_text:           The transcribed user speech (original, unmodified).
            history:             Previous conversation turns [{role, content}] for LLM context.
            asr_metadata:        Whisper acoustic confidence metadata for ASR protection.
            user_memory_context: Persistent multi-session memory facts from the Learning Vault.

        Returns:
            (response_dict, duration_ms)
        """
        start = time.perf_counter()
        settings = get_settings()
        history = history or []
        asr_metadata = asr_metadata or {}

        # --- ASR uncertainty detection & prefixing ---
        avg_lp = asr_metadata.get("avg_logprob")
        nsp = asr_metadata.get("no_speech_prob")
        is_globally_uncertain = asr_metadata.get("is_globally_uncertain", False)
        if not is_globally_uncertain:
            is_globally_uncertain = (
                (avg_lp is not None and avg_lp < -0.85) or
                (nsp is not None and nsp > 0.40)
            )

        uncertain_segments = asr_metadata.get("uncertain_segments", [])
        uncertain_segments = [s.strip() for s in uncertain_segments if s and s.strip()]

        if is_globally_uncertain:
            effective_text = f"[UNCERTAIN AUDIO] {user_text}"
        elif uncertain_segments:
            seg_list_repr = ", ".join(f'"{s}"' for s in uncertain_segments)
            effective_text = f"[UNCERTAIN SEGMENTS: {seg_list_repr}] {user_text}"
        else:
            effective_text = user_text

        # --- Provider selection ---
        # Primary: Gemini (if key is set)
        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip():
            try:
                result, duration_ms = await self._call_gemini(
                    effective_text, history, settings, start, raw_user_text=user_text, asr_metadata=asr_metadata,
                    user_memory_context=user_memory_context
                )
                if is_globally_uncertain:
                    result["corrections"] = []
                    result["spoken_text"] = result["response_text"]
                return result, duration_ms
            except Exception as e:
                print(f"[LLM Service] Gemini failed ({e}). Falling back to Groq...")

        # Primary / Fallback: Groq Qwen (AsyncGroq)
        if settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip():
            result, duration_ms = await self._call_groq(
                effective_text, history, settings, start, raw_user_text=user_text, asr_metadata=asr_metadata,
                user_memory_context=user_memory_context
            )
            if is_globally_uncertain:
                result["corrections"] = []
                result["spoken_text"] = result["response_text"]
            return result, duration_ms

        raise ValueError("Neither GEMINI_API_KEY nor GROQ_API_KEY is configured in .env.")

    async def _call_groq(
        self,
        user_text: str,
        history: list,
        settings,
        start: float,
        raw_user_text: str = "",
        asr_metadata: dict = None,
        user_memory_context: str = ""
    ) -> tuple[dict, int]:
        groq_client = self._get_async_groq(settings.GROQ_API_KEY.strip())

        messages = [{"role": "system", "content": FRIEND_SYSTEM_PROMPT}]
        if user_memory_context and user_memory_context.strip():
            messages.append({"role": "system", "content": user_memory_context.strip()})
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        
        # Crisp audit instruction on latest turn to enforce correction-first audit + conversational reply + JSON
        user_content = (
            f"{user_text}\n\n"
            "[MANDATORY AUDIT: Check for genuine mistakes (2 to 5 words max per card, at most 2-3 cards). "
            "Reply warmly and conversationally in response_text without grammar lectures. Output valid JSON only.]"
        )
        messages.append({"role": "user", "content": user_content})

        target_user_text = raw_user_text or user_text

        model_name = settings.GROQ_LLM_MODEL
        is_openai_model = "openai" in model_name.lower()
        allocated_tokens = _determine_max_tokens(user_text, is_openai_model)

        for attempt in range(4):
            try:
                call_kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": allocated_tokens,
                    "temperature": 0.3,
                }
                if is_openai_model:
                    call_kwargs["reasoning_effort"] = "low"
                else:
                    call_kwargs["response_format"] = {"type": "json_object"}

                response = await groq_client.chat.completions.create(**call_kwargs)
                raw = response.choices[0].message.content.strip()
                duration_ms = int((time.perf_counter() - start) * 1000)
                return _parse_llm_json(raw, target_user_text, asr_metadata), duration_ms
            except Exception as e:
                err_msg = str(e).lower()
                is_rate_limit = "rate_limit" in err_msg or "429" in err_msg or (hasattr(e, "status_code") and e.status_code == 429)

                # Defensive recovery: if Groq JSON mode rejected truncated generation, check failed_generation
                failed_gen = None
                if hasattr(e, "body") and isinstance(e.body, dict):
                    failed_gen = e.body.get("error", {}).get("failed_generation")
                if failed_gen and "max completion tokens" not in str(failed_gen).lower() and len(failed_gen) > 20:
                    duration_ms = int((time.perf_counter() - start) * 1000)
                    return _parse_llm_json(failed_gen, target_user_text, asr_metadata), duration_ms

                # If json_validate_failed or 400 Bad Request, retry without response_format constraint
                if "json_validate_failed" in err_msg or "bad request" in err_msg or (hasattr(e, "status_code") and e.status_code == 400):
                    try:
                        retry_resp = await groq_client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            max_tokens=allocated_tokens,
                            temperature=0.3
                        )
                        raw = retry_resp.choices[0].message.content.strip()
                        duration_ms = int((time.perf_counter() - start) * 1000)
                        return _parse_llm_json(raw, target_user_text, asr_metadata), duration_ms
                    except Exception as retry_err:
                        err_msg = str(retry_err).lower()
                        is_rate_limit = "rate_limit" in err_msg or "429" in err_msg or (hasattr(retry_err, "status_code") and retry_err.status_code == 429)

                # If rate-limited (e.g. daily token quota reached), fall back to alternate Groq models
                if is_rate_limit:
                    for fb_model in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
                        if fb_model != model_name:
                            try:
                                fb_kwargs = {
                                    "model": fb_model,
                                    "messages": messages,
                                    "max_tokens": _determine_max_tokens(user_text, is_openai=("openai" in fb_model)),
                                    "temperature": 0.3,
                                }
                                if "openai" in fb_model:
                                    fb_kwargs["reasoning_effort"] = "low"
                                fb_resp = await groq_client.chat.completions.create(**fb_kwargs)
                                raw = fb_resp.choices[0].message.content.strip()
                                duration_ms = int((time.perf_counter() - start) * 1000)
                                return _parse_llm_json(raw, target_user_text, asr_metadata), duration_ms
                            except Exception:
                                continue

                    if attempt < 3:
                        backoff = (attempt + 1) * 5  # 5s, 10s, 15s backoff
                        await asyncio.sleep(backoff)
                        continue

                raise e

    async def _call_gemini(
        self,
        user_text: str,
        history: list,
        settings,
        start: float,
        raw_user_text: str = "",
        asr_metadata: dict = None,
        user_memory_context: str = ""
    ) -> tuple[dict, int]:
        gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY.strip())

        # Build conversation context as plain text (Gemini non-standard path)
        contents = []
        for turn in history:
            contents.append(f"{turn['role'].capitalize()}: {turn['content']}")
        gemini_audit_instruction = (
            f"User: {user_text}\n"
            "[MANDATORY AUDIT: Check for genuine mistakes (2 to 5 words max per card, at most 2-3 cards). "
            "Reply warmly and conversationally in response_text without grammar lectures. Output valid JSON only.]"
        )
        contents.append(gemini_audit_instruction)

        target_user_text = raw_user_text or user_text

        system_instruction = FRIEND_SYSTEM_PROMPT
        if user_memory_context and user_memory_context.strip():
            system_instruction = f"{FRIEND_SYSTEM_PROMPT}\n\n{user_memory_context.strip()}"

        response = gemini_client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents="\n".join(contents),
            config={"system_instruction": system_instruction}
        )
        raw = response.text.strip()
        duration_ms = int((time.perf_counter() - start) * 1000)
        return _parse_llm_json(raw, target_user_text, asr_metadata), duration_ms
