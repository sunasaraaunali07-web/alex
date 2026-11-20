import asyncio
import time
import edge_tts

async def benchmark_edge_tts():
    # Test different texts and rates
    text_short = "In that sentence, say 'I went' instead of 'I go'."
    text_medium = "That sounds really frustrating. It is hard to explain yourself when someone keeps interrupting you. Did you manage to calm him down?"
    
    # 1. Single short
    t0 = time.perf_counter()
    c = edge_tts.Communicate(text_short, "en-US-JennyNeural")
    b = bytearray()
    async for chunk in c.stream():
        if chunk["type"] == "audio":
            b.extend(chunk["data"])
    d1 = int((time.perf_counter() - t0) * 1000)
    print(f"Short text ({len(text_short)} chars): {d1}ms | bytes: {len(b)}")

    # 2. Single medium
    t0 = time.perf_counter()
    c = edge_tts.Communicate(text_medium, "en-US-JennyNeural")
    b = bytearray()
    async for chunk in c.stream():
        if chunk["type"] == "audio":
            b.extend(chunk["data"])
    d2 = int((time.perf_counter() - t0) * 1000)
    print(f"Medium text ({len(text_medium)} chars): {d2}ms | bytes: {len(b)}")

if __name__ == "__main__":
    asyncio.run(benchmark_edge_tts())
