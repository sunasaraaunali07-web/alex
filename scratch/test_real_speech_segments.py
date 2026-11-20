import asyncio
import io
import edge_tts
from groq import Groq
from app.core.config import get_settings

async def test_real_speech_segments():
    # Generate real speech using edge-tts
    text = "Yesterday I went to college to meet my professor. We discussed the new research project for an hour."
    communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
    audio_data = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])
            
    print(f"Generated {len(audio_data)} bytes of real speech.")
    
    settings = get_settings()
    client = Groq(api_key=settings.GROQ_API_KEY.strip())
    file_tuple = ("speech.mp3", io.BytesIO(audio_data), "audio/mp3")
    
    res = client.audio.transcriptions.create(
        file=file_tuple,
        model=settings.GROQ_STT_MODEL,
        response_format="verbose_json"
    )
    d = res.model_dump()
    print("Full transcript text:", d.get("text"))
    print(f"Number of segments: {len(d.get('segments', []))}")
    for idx, seg in enumerate(d.get("segments", [])):
        print(f"\n--- Segment {idx} ---")
        print(f"  Time: {seg.get('start')}s - {seg.get('end')}s")
        print(f"  Text: '{seg.get('text')}'")
        print(f"  avg_logprob: {seg.get('avg_logprob')}")
        print(f"  no_speech_prob: {seg.get('no_speech_prob')}")
        print(f"  compression_ratio: {seg.get('compression_ratio')}")

if __name__ == "__main__":
    asyncio.run(test_real_speech_segments())
