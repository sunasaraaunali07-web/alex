import asyncio
import time
import edge_tts

async def synthesize_segment(text: str, voice: str = "en-US-JennyNeural", rate: str = "-6%"):
    c = edge_tts.Communicate(text, voice, rate=rate)
    b = bytearray()
    async for chunk in c.stream():
        if chunk["type"] == "audio":
            b.extend(chunk["data"])
    return bytes(b)

async def test_parallel():
    start = time.perf_counter()
    phrases = [
        "I go...",
        "I WENT...",
        "My friends was...",
        "My friends WERE...",
        "That sounds like a productive start! What is the main idea behind your project?"
    ]
    tasks = [synthesize_segment(p) for p in phrases]
    results = await asyncio.gather(*tasks)
    total_ms = int((time.perf_counter() - start) * 1000)
    print(f"Synthesized {len(results)} segments in {total_ms}ms")
    for idx, r in enumerate(results):
        print(f"  Segment {idx}: {len(r)} bytes")

if __name__ == "__main__":
    asyncio.run(test_parallel())
