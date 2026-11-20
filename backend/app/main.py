import asyncio
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.services.stt_service import STTService
from app.services.llm_service import LLMService
from app.services.tts_service import TTSService
from app.services.session_manager import session_manager
from app.services.memory_service import memory_service
from app.core.config import get_settings

app = FastAPI(title="AI English-Speaking Friend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stt_service = STTService()
llm_service = LLMService()
tts_service = TTSService()


@app.get("/api/health")
async def health_check():
    settings = get_settings()
    has_groq = bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip())
    has_gemini = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip())
    return {
        "status": "ok",
        "service": "AI English-Speaking Friend",
        "groq_configured": has_groq,
        "gemini_configured": has_gemini
    }


@app.post("/api/turn")
async def process_turn(
    audio: UploadFile = File(...),
    session_id: str = Form(None),
    user_id: str = Form(None)
):
    """
    Processes one full conversational turn:
      1. STT — transcribes user audio
      2. Memory — retrieves persistent user profile, grammar history, and learning context
      3. LLM — generates calibrated response with corrections + reply + remembered context
      4. TTS — synthesises the spoken_text (corrections woven in naturally + reply)
      5. Session & Vault — stores live dialogue and updates persistent memory asynchronously
    """
    audio_bytes = await audio.read()

    # 0. Session & User tracking
    uid = user_id or "default_user"
    session = session_manager.get_or_create(session_id, user_id=uid)
    sid = session.session_id
    uid = session.user_id

    try:
        # 1. Speech-to-Text with acoustic confidence metadata
        transcript, stt_ms, asr_metadata = await stt_service.transcribe(
            audio_bytes, audio.filename or "input.webm"
        )

        # Handle empty, silent or failed speech recognition gracefully without breaking the conversation
        if not transcript or not transcript.strip() or transcript.strip() in [".", "...", "?", "!"]:
            empty_msg = "I didn't quite catch that. Could you please say that again?"
            tts_result = await tts_service.synthesize_turn(
                response_text=empty_msg,
                corrections=[],
                spoken_text=empty_msg
            )
            return {
                "session_id": sid,
                "user_id": uid,
                "user_transcript": "",
                "corrections": [],
                "response_text": empty_msg,
                "spoken_text": empty_msg,
                "speech_segments": tts_result.get("speech_segments", []),
                "timing_schedule": tts_result.get("timing_schedule", []),
                "audio_base64": tts_result.get("audio_base64"),
                "tts_engine": tts_result.get("tts_engine", "edge-tts"),
                "latency_ms": {
                    "stt": stt_ms,
                    "llm": 0,
                    "tts": tts_result.get("tts_ms", 0),
                    "total": stt_ms + tts_result.get("tts_ms", 0)
                }
            }

        # 2. Retrieve persistent learning memory context from Vault (including exact verbatim history if requested)
        user_memory_context = memory_service.get_user_memory_context(uid, current_query=transcript)

        # 3. Structured LLM response: corrections + conversational reply + memory awareness
        history = session_manager.get_dialogue_history_for_llm(sid, window=30)
        llm_result, llm_ms = await llm_service.generate_response(
            transcript,
            history=history,
            asr_metadata=asr_metadata,
            user_memory_context=user_memory_context
        )

        corrections = llm_result.get("corrections", [])
        response_text = llm_result.get("response_text", "")
        spoken_text = llm_result.get("spoken_text", response_text)

        # 4. TTS — synthesise segmented speech with adaptive pauses + unified audio
        tts_result = await tts_service.synthesize_turn(
            response_text=response_text,
            corrections=corrections,
            spoken_text=spoken_text
        )
        audio_b64 = tts_result.get("audio_base64")
        speech_segments = tts_result.get("speech_segments", [])
        timing_schedule = tts_result.get("timing_schedule", [])
        tts_engine = tts_result.get("tts_engine", "edge-tts")
        tts_ms = tts_result.get("tts_ms", 0)

        # 5. Store the turn in session manager & persist to permanent SQLite conversation archive
        session_manager.add_turn(sid, transcript, response_text, asr_metadata, corrections)
        turn_index = len(session.full_transcript) // 2
        memory_service.record_message(sid, uid, "user", transcript, turn_index)
        memory_service.record_message(sid, uid, "assistant", response_text, turn_index)

        # 6. Asynchronous, non-blocking turn memory extraction (zero latency impact)
        asyncio.create_task(
            asyncio.to_thread(
                memory_service.extract_turn_memories,
                uid,
                sid,
                transcript,
                response_text,
                corrections
            )
        )

        total_ms = stt_ms + llm_ms + tts_ms

        return {
            "session_id": sid,
            "user_id": uid,
            "user_transcript": transcript,
            "corrections": corrections,
            "response_text": response_text,
            "spoken_text": spoken_text,
            "speech_segments": speech_segments,
            "timing_schedule": timing_schedule,
            "audio_base64": audio_b64,
            "tts_engine": tts_engine,
            "latency_ms": {
                "stt": stt_ms,
                "llm": llm_ms,
                "tts": tts_ms,
                "total": total_ms
            }
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")


@app.get("/api/user/{user_id}/memory")
async def get_user_memory(user_id: str):
    """Retrieves the complete persistent Learning Memory Vault for a given user."""
    return memory_service.get_all_user_memory(user_id)


# Mount static frontend files
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
