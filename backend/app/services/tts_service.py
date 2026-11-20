import asyncio
import time
import base64
from typing import List, Dict, Any, Optional
import edge_tts
from app.core.config import get_settings
from app.services.pause_manager import build_timing_schedule

class TTSService:
    async def synthesize(self, text: str, voice: Optional[str] = None, rate: str = "-6%", pitch: str = "+0Hz") -> tuple[str | None, str, int]:
        """
        Synthesizes text to audio via Edge-TTS.
        Returns (audio_base64 or None, tts_engine_name, duration_ms).
        Guaranteed to not crash — returns None on failure so browser fallback triggers.
        """
        start = time.perf_counter()
        settings = get_settings()
        v = voice or settings.DEFAULT_TTS_VOICE
        try:
            communicate = edge_tts.Communicate(text, v, rate=rate, pitch=pitch)
            audio_data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
            
            duration_ms = int((time.perf_counter() - start) * 1000)
            b64_audio = base64.b64encode(audio_data).decode("utf-8")
            return b64_audio, "edge-tts", duration_ms
        except Exception as e:
            print(f"[TTS Service] Edge-TTS synthesis failed ({e}). Signaling browser fallback.")
            duration_ms = int((time.perf_counter() - start) * 1000)
            return None, "browser-fallback", duration_ms

    async def _synthesize_bytes(self, text: str, voice: str, rate: str = "-6%", pitch: str = "+0Hz") -> bytes | None:
        """Helper to synthesize a single segment to raw bytes."""
        try:
            c = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            b = bytearray()
            async for chunk in c.stream():
                if chunk["type"] == "audio":
                    b.extend(chunk["data"])
            return bytes(b)
        except Exception as ex:
            print(f"[TTS Segment Error] Failed for '{text}': {ex}")
            return None

    async def synthesize_turn(
        self,
        response_text: str,
        corrections: List[Dict[str, Any]],
        spoken_text: str,
        voice: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes the turn into segmented speech units with calculated adaptive pauses
        AND provides a unified audio_base64 fallback.
        """
        start = time.perf_counter()
        settings = get_settings()
        v = voice or settings.DEFAULT_TTS_VOICE
        
        # Build adaptive timing schedule
        timing_schedule = build_timing_schedule(corrections)
        
        # Tasks for concurrent synthesis
        segment_tasks = []
        segment_metadata = []

        if corrections:
            # Cap spoken comparisons to top 2 genuine errors to keep feedback punchy and fast
            spoken_corrections = corrections[:2]
            for idx, c in enumerate(spoken_corrections):
                spoken_comp = c.get("spoken_comparison")
                orig = str(c.get("original", "")).strip().strip("'\"")
                corr = str(c.get("corrected", "")).strip().strip("'\"")

                if not spoken_comp or (orig and orig.lower() not in spoken_comp.lower()) or (corr and corr.lower() not in spoken_comp.lower()):
                    if idx == 0:
                        spoken_comp = f"In that sentence, say '{corr}' instead of '{orig}'."
                    else:
                        spoken_comp = f"Also, say '{corr}' instead of '{orig}'."
                else:
                    import re
                    spoken_comp = re.sub(r"['\"]\s*,\s*(?:because|since|due to)\b.*$", "'", spoken_comp, flags=re.IGNORECASE).strip().rstrip(".,;:")
                    spoken_comp = f"{spoken_comp}."

                sched = timing_schedule[idx] if idx < len(timing_schedule) else {}
                pause_after = sched.get("pause_after_correction_ms", 750)

                # Synthesize the explicit spoken comparison segment (+4% keeps lively natural conversational tempo)
                segment_tasks.append(self._synthesize_bytes(spoken_comp, v, rate="+4%", pitch="+0Hz"))
                segment_metadata.append({
                    "segment_id": f"corr_{idx}",
                    "type": "correction",
                    "text": spoken_comp,
                    "pause_after_ms": pause_after,
                    "card_index": idx,
                    "emphasized_word": c.get("emphasized_word", "")
                })

        # Conversational response (+3% natural conversational tempo)
        segment_tasks.append(self._synthesize_bytes(response_text, v, rate="+3%", pitch="+0Hz"))
        segment_metadata.append({
            "segment_id": "response",
            "type": "response",
            "text": response_text,
            "pause_after_ms": 0,
            "card_index": None
        })

        # Execute all segments in parallel
        segment_results = await asyncio.gather(*segment_tasks)

        # Assemble speech segments
        speech_segments = []
        unified_audio_data = bytearray()

        for meta, audio_bytes in zip(segment_metadata, segment_results):
            if audio_bytes:
                b64 = base64.b64encode(audio_bytes).decode("utf-8")
                unified_audio_data.extend(audio_bytes)
            else:
                b64 = None
            
            speech_segments.append({
                **meta,
                "audio_base64": b64
            })

        duration_ms = int((time.perf_counter() - start) * 1000)
        unified_b64 = base64.b64encode(unified_audio_data).decode("utf-8") if unified_audio_data else None

        return {
            "audio_base64": unified_b64,
            "speech_segments": speech_segments,
            "timing_schedule": timing_schedule,
            "tts_engine": "edge-tts" if unified_b64 else "browser-fallback",
            "tts_ms": duration_ms
        }
