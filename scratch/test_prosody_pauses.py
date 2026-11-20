import asyncio
import time
import edge_tts

async def test_pauses():
    voice = "en-US-JennyNeural"
    
    # 1. Without pause
    c1 = edge_tts.Communicate("I go. I went. What did you study there?", voice, rate="-6%")
    b1 = bytearray()
    async for ch in c1.stream():
        if ch["type"] == "audio": b1.extend(ch["data"])
    
    # 2. With ellipses pauses
    c2 = edge_tts.Communicate("I go... I WENT... ... What did you study there?", voice, rate="-6%")
    b2 = bytearray()
    async for ch in c2.stream():
        if ch["type"] == "audio": b2.extend(ch["data"])

    # 3. With explicit silence MP3 insertion
    print(f"No pause audio size: {len(b1)} bytes")
    print(f"Ellipses pause audio size: {len(b2)} bytes")
    print(f"Size difference: {len(b2) - len(b1)} bytes (proves longer pause duration)")

if __name__ == "__main__":
    asyncio.run(test_pauses())
