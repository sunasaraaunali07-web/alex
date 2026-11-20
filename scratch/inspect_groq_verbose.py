import asyncio
import io
import wave
import struct
import math
from groq import Groq
from app.core.config import get_settings

def create_synthetic_audio():
    # 2 seconds of 440Hz sine wave tone
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        for i in range(16000 * 2):
            sample = int(32767.0 * 0.1 * math.sin(2.0 * math.pi * 440.0 * i / 16000))
            wav.writeframes(struct.pack('<h', sample))
    return buf.getvalue()

async def inspect_groq_verbose():
    settings = get_settings()
    client = Groq(api_key=settings.GROQ_API_KEY.strip())
    audio_bytes = create_synthetic_audio()
    
    file_tuple = ("test.wav", io.BytesIO(audio_bytes), "audio/wav")
    try:
        res = client.audio.transcriptions.create(
            file=file_tuple,
            model=settings.GROQ_STT_MODEL,
            response_format="verbose_json"
        )
        print("Type of res:", type(res))
        print("res attributes:", dir(res))
        if hasattr(res, 'model_dump'):
            d = res.model_dump()
            print("Keys in model_dump:", list(d.keys()))
            if "segments" in d and d["segments"]:
                print("First segment keys:", list(d["segments"][0].keys()))
                print("First segment sample:", d["segments"][0])
        elif isinstance(res, dict):
            print("res keys:", list(res.keys()))
        else:
            print("res dict:", getattr(res, '__dict__', {}))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(inspect_groq_verbose())
