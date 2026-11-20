import asyncio
import time
import os
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))
os.environ["PYTHONPATH"] = str(backend_dir)

from app.services.tts_service import TTSService

async def test_single_call():
    tts = TTSService()
    full_text = (
        "In that sentence, say 'he doesn't understand' instead of 'he don't understand'. ... "
        "Also, say 'starts arguing' instead of 'start arguing with'. ... "
        "That sounds really frustrating. It is hard to explain yourself when someone keeps interrupting you. "
        "Did you manage to calm him down, or did the argument continue?"
    )
    voice = "en-US-JennyNeural"

    t0 = time.perf_counter()
    b = await tts._synthesize_bytes(full_text, voice)
    dur = int((time.perf_counter() - t0) * 1000)
    print(f"Single Edge-TTS call for FULL text ({len(full_text)} chars): {dur}ms | Audio bytes: {len(b) if b else 0}")

if __name__ == "__main__":
    asyncio.run(test_single_call())
