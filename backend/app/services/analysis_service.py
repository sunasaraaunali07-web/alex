import json
import re
from typing import List, Dict, Any
from groq import AsyncGroq
from google import genai
from app.core.config import get_settings

# ---------------------------------------------------------------------------
# Chunk Analysis Prompt
# Focus: identify patterns in a conversation section, knowing real-time
# corrections were already given after each user turn.
# ---------------------------------------------------------------------------
CHUNK_ANALYSIS_SYSTEM_PROMPT = """
You are an expert English language mentor analyzing a section of a spoken English conversation.

IMPORTANT CONTEXT: The user received real-time grammar corrections after each of their messages during the conversation.
These corrections are shown in [CORRECTIONS GIVEN: ...] markers beside each user turn.

YOUR GOAL: Identify RECURRING error patterns — mistakes the user made more than once.

WHAT TO DO:
1. Look at ALL user utterances (original speech, before any corrections).
2. Identify the CATEGORY of each error (e.g. "Past-Tense Verb Usage").
3. Note which errors were already corrected in real-time vs. entirely new ones.
4. Note speaking fluency patterns (filler words, sentence restarts, pacing).
5. Note vocabulary Alex used that the user might find useful to learn.

CRITICAL RULES:
1. ASR PROTECTION: Do NOT analyze turns marked [UNCERTAIN AUDIO]. Skip them entirely.
2. ERROR TYPES — be precise:
   - "grammar": wrong verb tense, subject-verb disagreement, wrong/missing auxiliary, broken clauses
   - "expression": grammatically acceptable but unnatural due to Hindi/Gujarati translation
   Do NOT flag a grammatically correct sentence as an error.
3. ACCURACY OVER QUANTITY: Return errors: [] if the user made no genuine mistakes in this section.
4. Do NOT invent errors just to fill the response.

OUTPUT FORMAT — respond strictly with valid JSON only:
{
  "section_summary": "1–2 sentence description of what the user talked about in this section.",
  "errors_found": [
    {
      "original_sentence": "Exact user utterance containing the error.",
      "error_category": "Past-Tense Verb Usage",
      "error_type": "grammar",
      "already_corrected_realtime": true
    }
  ],
  "speaking_patterns": [
    "Observation about filler words, restarts, pacing — or 'Clear, continuous delivery.'"
  ],
  "alex_vocabulary_used": [
    {
      "word": "Notable word or phrase Alex actually said in this section.",
      "context": "Brief quote showing how Alex used it."
    }
  ]
}
"""

# ---------------------------------------------------------------------------
# Consolidation Prompt
# Focus: recurring patterns across the whole conversation, not individual repeats.
# ---------------------------------------------------------------------------
CONSOLIDATION_SYSTEM_PROMPT = """
You are an expert English language coach producing the final learning report for an English learner who just finished a practice conversation with their AI friend Alex.

IMPORTANT: The user already received real-time grammar corrections after EVERY message during the conversation.
This final report must NOT simply re-list those individual corrections.

FOCUS ON:
1. RECURRING PATTERNS — the same type of mistake the user made 2 or more times.
2. KEY LEARNING AREAS — the most important grammar concepts for this user to practise.
3. ACTIONABLE NEXT STEPS — 2–3 specific, practical things to practise.

RULES:
- A category is "recurring" ONLY if the same type of error appeared 2+ times. occurrence_count must be accurate.
- If no recurring mistakes exist, return "recurring_mistakes": [] — do NOT invent patterns.
- "examples" in recurring_mistakes must be actual user sentences from the data, not invented ones.
- Do NOT repeat corrections already well-covered in real-time just to fill the report.
- Do NOT invent vocabulary Alex didn't actually use.
- Do NOT generate false positives or vague praise without evidence.

OUTPUT FORMAT — respond strictly with valid JSON only:
{
  "conversational_overview": "Warm, encouraging 2-sentence summary of what the user talked about.",
  "fluency_celebration": "1–2 sentences highlighting what the user communicated well.",
  "recurring_mistakes": [
    {
      "category": "Past-Tense Verb Usage",
      "occurrence_count": 3,
      "underlying_pattern": "Using present-tense verbs when describing past events.",
      "examples": ["Yesterday I go to market", "Last week I meet him"]
    }
  ],
  "speaking_patterns": "Observations on fluency, filler words, or confirmation of clear, easy-to-follow speech.",
  "useful_vocabulary": [
    {
      "word": "Word or phrase Alex actually used",
      "context_used": "Quote showing how Alex used it in conversation"
    }
  ],
  "practice_advice": "2–3 specific, practical, encouraging practice recommendations for the user's next session."
}
"""


class AnalysisService:
    def __init__(self):
        self._async_groq_client = None

    def _get_async_groq(self, api_key: str) -> AsyncGroq:
        """Lazy-initialised, cached AsyncGroq client."""
        if not self._async_groq_client:
            self._async_groq_client = AsyncGroq(api_key=api_key)
        return self._async_groq_client

    async def analyze_session(self, full_transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes the complete session transcript for recurring language patterns.

        Pipeline:
        1. Count user turns (short-circuit if too few).
        2. Chunk the dialogue into sections (max 8 user turns each).
        3. Analyze each chunk for error patterns and vocabulary.
        4. Consolidate all chunk findings into ONE cohesive recurring-pattern report.
        """
        settings = get_settings()
        if not settings.GROQ_API_KEY and not settings.GEMINI_API_KEY:
            raise ValueError("No API key available for conversation analysis.")

        # Count user turns and check minimum length
        user_turn_count = sum(1 for m in full_transcript if m.get("role") == "user" and m.get("content", "").strip())

        if user_turn_count < 2:
            return {
                "status": "too_short",
                "turn_count": user_turn_count,
                "conversational_overview": "We had a quick greeting today.",
                "fluency_celebration": (
                    "Thanks for speaking with Alex! Have a slightly longer conversation "
                    "(at least 2–3 turns) next time to receive a full learning report."
                ),
                "recurring_mistakes": [],
                "speaking_patterns": "Conversation was too brief to evaluate speaking patterns.",
                "useful_vocabulary": [],
                "practice_advice": "Try sharing a short story about your day or your work in the next chat!"
            }

        # Chunk the dialogue for analysis
        chunks = self._chunk_dialogue(full_transcript, max_user_turns_per_chunk=8)

        chunk_results = []
        for idx, chunk in enumerate(chunks):
            chunk_text = "\n".join(chunk)
            chunk_data = await self._analyze_chunk(chunk_text, idx + 1, len(chunks))
            if chunk_data:
                chunk_results.append(chunk_data)

        # Consolidate into ONE final report
        final_report = await self._consolidate_report(chunk_results, full_transcript, user_turn_count)
        return final_report

    def _chunk_dialogue(
        self,
        full_transcript: List[Dict[str, Any]],
        max_user_turns_per_chunk: int = 8
    ) -> List[List[str]]:
        """
        Divides the transcript into chronological chunks.
        Each user turn line includes:
          - The original user text
          - [UNCERTAIN AUDIO] flag if ASR confidence was low
          - [CORRECTIONS GIVEN: ...] showing what was corrected in real-time
        """
        chunks = []
        current_chunk: List[str] = []
        current_user_count = 0

        for m in full_transcript:
            role = m.get("role", "")
            content = m.get("content", "").strip()
            if not content:
                continue

            if role == "user":
                # When previous chunk reached max capacity, seal it on the new user turn
                # so the prior exchange's assistant reply stays in the same chunk.
                if current_user_count >= max_user_turns_per_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_user_count = 0

                # Build annotated user line
                line = f'User: "{content}"'

                # ASR confidence gate
                asr = m.get("asr_metadata", {})
                avg_lp = asr.get("avg_logprob")
                nsp = asr.get("no_speech_prob")
                is_low_conf = (
                    (avg_lp is not None and avg_lp < -0.85) or
                    (nsp is not None and nsp > 0.40)
                )
                if is_low_conf:
                    line += " [UNCERTAIN AUDIO]"

                # Real-time correction context
                corrections = m.get("corrections", [])
                corr_parts = []
                for c in corrections:
                    orig = c.get("original", "").strip()
                    corr = c.get("corrected", "").strip()
                    if orig and corr and orig.lower() != corr.lower():
                        corr_parts.append(f'"{orig}" → "{corr}"')
                if corr_parts:
                    line += f" [CORRECTIONS GIVEN: {'; '.join(corr_parts)}]"

                current_chunk.append(line)
                current_user_count += 1

            else:  # assistant turn
                current_chunk.append(f'Alex: "{content}"')

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [['User: "Hello"', 'Alex: "Hi there!"']]

    async def _analyze_chunk(
        self,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int
    ) -> Dict[str, Any]:
        """Analyzes one chronological section of dialogue for error patterns."""
        settings = get_settings()
        user_prompt = (
            f"Here is Section {chunk_index} of {total_chunks} from the conversation:\n\n"
            f"--- DIALOGUE SECTION ---\n{chunk_text}\n--- END DIALOGUE SECTION ---\n\n"
            "Analyze this section according to the instructions. Output strictly valid JSON."
        )

        try:
            if settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip():
                groq_client = self._get_async_groq(settings.GROQ_API_KEY.strip())
                response = await groq_client.chat.completions.create(
                    model=settings.GROQ_LLM_MODEL,
                    messages=[
                        {"role": "system", "content": CHUNK_ANALYSIS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                raw_json = response.choices[0].message.content.strip()
                return json.loads(raw_json)
        except Exception as e:
            print(f"[Analysis Error] Chunk {chunk_index} failed: {e}")
            return {}

    async def _consolidate_report(
        self,
        chunk_results: List[Dict[str, Any]],
        full_transcript: List[Dict[str, Any]],
        user_turn_count: int
    ) -> Dict[str, Any]:
        """Synthesises chunk findings into ONE unified recurring-pattern report."""
        settings = get_settings()

        # Collect raw findings across all chunks
        all_errors: List[Dict[str, Any]] = []
        all_vocab: List[Dict[str, Any]] = []
        all_speaking_patterns: List[str] = []

        for res in chunk_results:
            all_errors.extend(res.get("errors_found", []))
            all_vocab.extend(res.get("alex_vocabulary_used", []))
            all_speaking_patterns.extend(res.get("speaking_patterns", []))

        consolidation_input = {
            "total_user_turns": user_turn_count,
            "all_errors_across_conversation": all_errors,
            "detected_speaking_patterns": all_speaking_patterns,
            "candidate_vocabulary_used_by_alex": all_vocab
        }

        user_prompt = (
            "Here is the aggregated analysis data from all conversation sections:\n\n"
            + json.dumps(consolidation_input, indent=2)
            + "\n\nCreate the final consolidated report in the specified JSON format. "
            "Only include recurring_mistakes for categories that genuinely appeared 2+ times. "
            "Ensure occurrence_count values are accurate."
        )

        try:
            if settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip():
                groq_client = self._get_async_groq(settings.GROQ_API_KEY.strip())
                response = await groq_client.chat.completions.create(
                    model=settings.GROQ_LLM_MODEL,
                    messages=[
                        {"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=2500,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                raw_json = response.choices[0].message.content.strip()
                final_data = json.loads(raw_json)

                final_data["status"] = "completed"
                final_data["turn_count"] = user_turn_count

                # Verify vocabulary was actually spoken by Alex
                alex_full_text = " ".join(
                    m["content"].lower()
                    for m in full_transcript
                    if m.get("role") == "assistant"
                )
                verified_vocab = []
                for v in final_data.get("useful_vocabulary", []):
                    word = v.get("word", "").lower()
                    if word and any(term in alex_full_text for term in word.split()):
                        verified_vocab.append(v)
                # Keep verified; fall back to top-3 candidates if none verified
                final_data["useful_vocabulary"] = (
                    verified_vocab if verified_vocab
                    else final_data.get("useful_vocabulary", [])[:3]
                )

                return final_data

        except Exception as e:
            print(f"[Consolidation Error] Failed: {e}")

        # Deterministic fallback if consolidation LLM fails
        return self._fallback_consolidation(all_errors, all_vocab, user_turn_count)

    def _fallback_consolidation(
        self,
        errors: list,
        vocab: list,
        turn_count: int
    ) -> Dict[str, Any]:
        """Deterministic fallback if the consolidation LLM call fails."""
        # Group errors by category to find recurring ones
        cat_map: Dict[str, List[str]] = {}
        for err in errors:
            cat = err.get("error_category", "General Grammar")
            sentence = err.get("original_sentence", "")
            if sentence:
                cat_map.setdefault(cat, []).append(sentence)

        recurring = []
        for cat, examples in cat_map.items():
            if len(examples) >= 2:
                recurring.append({
                    "category": cat,
                    "occurrence_count": len(examples),
                    "underlying_pattern": f"Repeated issues with {cat.lower()}.",
                    "examples": examples[:3]
                })

        return {
            "status": "completed",
            "turn_count": turn_count,
            "conversational_overview": "We had a productive English conversation today.",
            "fluency_celebration": "You expressed your ideas clearly and kept the conversation flowing well.",
            "recurring_mistakes": recurring,
            "speaking_patterns": "Speech pacing was clear throughout the dialogue.",
            "useful_vocabulary": vocab[:3],
            "practice_advice": (
                "Focus on practising verb tenses and subject-verb agreement in your next session. "
                "Try narrating a short story about your day using past-tense sentences."
            )
        }


analysis_service = AnalysisService()
