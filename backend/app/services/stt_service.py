import io
import time
from groq import AsyncGroq
from app.core.config import get_settings

class STTService:
    def __init__(self):
        self._async_client = None

    def _get_client(self, api_key: str) -> AsyncGroq:
        if not self._async_client:
            self._async_client = AsyncGroq(api_key=api_key)
        return self._async_client

    async def transcribe(self, audio_bytes: bytes, filename: str = "input.webm") -> tuple[str, int, dict]:
        """
        Transcribes audio asynchronously using Groq Whisper.
        Returns:
            transcript (str): Clean transcribed text.
            duration_ms (int): Execution time in milliseconds.
            asr_metadata (dict): Defensively extracted acoustic confidence metrics (avg_logprob, no_speech_prob).
        """
        settings = get_settings()
        if not settings.GROQ_API_KEY or not settings.GROQ_API_KEY.strip():
            raise ValueError("GROQ_API_KEY is missing. Please add your key to the .env file.")
        
        client = self._get_client(settings.GROQ_API_KEY.strip())
        start = time.perf_counter()
        file_tuple = (filename, io.BytesIO(audio_bytes), "audio/webm")
        
        transcription = await client.audio.transcriptions.create(
            file=file_tuple,
            model=settings.GROQ_STT_MODEL,
            prompt="A natural English conversation with friendly, clear spoken everyday words.",
            language="en",
            response_format="verbose_json"
        )
        duration_ms = int((time.perf_counter() - start) * 1000)
        
        # Defensive extraction of Whisper acoustic metrics (avoids assuming fields exist)
        asr_metadata = {
            "avg_logprob": None,
            "min_logprob": None,
            "no_speech_prob": None,
            "has_segments": False,
            "segments": [],
            "uncertain_segments": [],
            "is_globally_uncertain": False
        }
        
        try:
            raw_segments = getattr(transcription, "segments", None)
            if raw_segments and isinstance(raw_segments, list) and len(raw_segments) > 0:
                asr_metadata["has_segments"] = True
                parsed_segments = []
                logprobs = []
                no_speeches = []
                uncertain_texts = []

                for idx, seg in enumerate(raw_segments):
                    # Support both Pydantic model objects and plain dictionaries
                    seg_dict = seg if isinstance(seg, dict) else getattr(seg, "model_dump", lambda: getattr(seg, "__dict__", {}))()
                    if not isinstance(seg_dict, dict):
                        continue
                    
                    seg_id = seg_dict.get("id", idx)
                    start_t = seg_dict.get("start", 0.0)
                    end_t = seg_dict.get("end", 0.0)
                    text = str(seg_dict.get("text", "")).strip()
                    
                    lp = seg_dict.get("avg_logprob")
                    nsp = seg_dict.get("no_speech_prob")
                    
                    lp_val = float(lp) if lp is not None else None
                    nsp_val = float(nsp) if nsp is not None else None
                    
                    if lp_val is not None:
                        logprobs.append(lp_val)
                    if nsp_val is not None:
                        no_speeches.append(nsp_val)
                    
                    # Segment is uncertain if avg_logprob is very low (< -0.85) or no_speech_prob is high (> 0.40)
                    is_seg_uncertain = (
                        (lp_val is not None and lp_val < -0.85) or
                        (nsp_val is not None and nsp_val > 0.40)
                    )
                    
                    if is_seg_uncertain and text:
                        uncertain_texts.append(text)
                    
                    parsed_segments.append({
                        "id": seg_id,
                        "start": start_t,
                        "end": end_t,
                        "text": text,
                        "avg_logprob": lp_val,
                        "no_speech_prob": nsp_val,
                        "is_uncertain": is_seg_uncertain
                    })
                
                asr_metadata["segments"] = parsed_segments
                asr_metadata["uncertain_segments"] = uncertain_texts

                if logprobs:
                    asr_metadata["avg_logprob"] = sum(logprobs) / len(logprobs)
                    asr_metadata["min_logprob"] = min(logprobs)
                if no_speeches:
                    asr_metadata["no_speech_prob"] = max(no_speeches)

                # Turn is globally uncertain if:
                # 1. Overall avg_logprob < -0.85
                # 2. Max no_speech_prob > 0.40
                # 3. All non-empty segments are uncertain
                is_global = False
                if asr_metadata["avg_logprob"] is not None and asr_metadata["avg_logprob"] < -0.85:
                    is_global = True
                elif asr_metadata["no_speech_prob"] is not None and asr_metadata["no_speech_prob"] > 0.40:
                    is_global = True
                elif parsed_segments and all(s.get("is_uncertain", False) for s in parsed_segments if s.get("text")):
                    is_global = True

                asr_metadata["is_globally_uncertain"] = is_global
        except Exception as e:
            # Fallback gracefully if Groq schema varies
            asr_metadata["extraction_error"] = str(e)
            
        transcript_text = transcription.text.strip() if hasattr(transcription, "text") else str(transcription).strip()
        return transcript_text, duration_ms, asr_metadata
