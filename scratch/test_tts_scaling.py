import asyncio
import time
import os
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))
os.environ["PYTHONPATH"] = str(backend_dir)

from app.services.tts_service import TTSService

async def test_tts_speeds():
    tts = TTSService()
    
    # Case 0: Only response
    t0 = time.perf_counter()
    r0 = await tts.synthesize_turn("Oh that sounds really frustrating! Did you manage to fix things with your friend?", [], "")
    print(f"0 corrections (1 segment): {int((time.perf_counter() - t0)*1000)}ms")

    # Case 1: 1 correction + response
    c1 = [{"original": "I go", "corrected": "I went", "spoken_comparison": "In that sentence, say 'I went' instead of 'I go'."}]
    t0 = time.perf_counter()
    r1 = await tts.synthesize_turn("Oh that sounds really frustrating! Did you manage to fix things with your friend?", c1, "")
    print(f"1 correction (2 segments): {int((time.perf_counter() - t0)*1000)}ms")

    # Case 2: 2 corrections + response
    c2 = c1 + [{"original": "because I have to", "corrected": "because I had to", "spoken_comparison": "Also, say 'because I had to' instead of 'because I have to'."}]
    t0 = time.perf_counter()
    r2 = await tts.synthesize_turn("Oh that sounds really frustrating! Did you manage to fix things with your friend?", c2, "")
    print(f"2 corrections (3 segments): {int((time.perf_counter() - t0)*1000)}ms")

if __name__ == "__main__":
    asyncio.run(test_tts_speeds())
