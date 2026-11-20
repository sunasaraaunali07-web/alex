import asyncio
import time
import os
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))
os.environ["PYTHONPATH"] = str(backend_dir)

from app.services.tts_service import TTSService

async def test_concurrency():
    tts = TTSService()
    texts = [
        "In that sentence, say 'I went' instead of 'I go'.",
        "Also, say 'because I had to' instead of 'because I have to'.",
        "Oh that sounds really frustrating! Did you manage to fix things with your friend?"
    ]
    voice = "en-US-JennyNeural"

    # Strategy 1: Current asyncio.gather (all at once)
    t0 = time.perf_counter()
    r1 = await asyncio.gather(*[tts._synthesize_bytes(t, voice) for t in texts])
    d1 = int((time.perf_counter() - t0) * 1000)
    print(f"Strategy 1 (gather all 3): {d1}ms")

    # Strategy 2: Sequential (one after another)
    t0 = time.perf_counter()
    r2 = []
    for t in texts:
        r2.append(await tts._synthesize_bytes(t, voice))
    d2 = int((time.perf_counter() - t0) * 1000)
    print(f"Strategy 2 (sequential 3): {d2}ms")

    # Strategy 3: Concurrency semaphore = 2
    sem = asyncio.Semaphore(2)
    async def limited(t):
        async with sem:
            return await tts._synthesize_bytes(t, voice)
    t0 = time.perf_counter()
    r3 = await asyncio.gather(*[limited(t) for t in texts])
    d3 = int((time.perf_counter() - t0) * 1000)
    print(f"Strategy 3 (semaphore 2): {d3}ms")

if __name__ == "__main__":
    asyncio.run(test_concurrency())
