import asyncio
import time
import edge_tts

async def test_rates():
    text = "In that sentence, say 'I went to college' instead of 'I go to college'."
    
    for rate in ["-6%", "+0%", "+5%"]:
        t0 = time.perf_counter()
        c = edge_tts.Communicate(text, "en-US-JennyNeural", rate=rate)
        b = bytearray()
        async for chunk in c.stream():
            if chunk["type"] == "audio":
                b.extend(chunk["data"])
        dur = int((time.perf_counter() - t0) * 1000)
        print(f"Rate {rate} -> {dur}ms | bytes: {len(b)}")

if __name__ == "__main__":
    asyncio.run(test_rates())
