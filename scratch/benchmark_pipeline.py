import asyncio
import time
import os
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))
os.environ["PYTHONPATH"] = str(backend_dir)

from app.services.stt_service import STTService
from app.services.llm_service import LLMService
from app.services.tts_service import TTSService

async def benchmark():
    stt = STTService()
    llm = LLMService()
    tts = TTSService()

    # Let's test with a representative user input
    test_text = "Yesterday I was going to college but suddenly my friend called me and said that he is waiting for me since 2 hours. I told him that I already reached there but actually I was still at home. After that he became angry because I didn't inform him earlier and I was not able to explain him what happened. I think he don't understand my situation because whenever I try to explain something he always interrupts me and start arguing with without listening completely."

    print("--- BENCHMARKING PIPELINE COMPONENTS ---")

    # 1. LLM Latency
    t0 = time.perf_counter()
    llm_res, llm_ms = await llm.generate_response(test_text)
    t1 = time.perf_counter()
    print(f"LLM time: {int((t1 - t0)*1000)}ms (internal: {llm_ms}ms)")
    print(f"  Corrections count: {len(llm_res.get('corrections', []))}")
    print(f"  Response text length: {len(llm_res.get('response_text', ''))}")

    # 2. TTS Latency
    corrections = llm_res.get("corrections", [])
    response_text = llm_res.get("response_text", "")
    t2 = time.perf_counter()
    tts_res = await tts.synthesize_turn(
        response_text=response_text,
        corrections=corrections,
        spoken_text=llm_res.get("spoken_text", response_text)
    )
    t3 = time.perf_counter()
    print(f"TTS time: {int((t3 - t2)*1000)}ms (internal: {tts_res.get('tts_ms')}ms)")
    print(f"  Segments count: {len(tts_res.get('speech_segments', []))}")

    # Total backend processing time (excluding STT for text benchmark):
    print(f"Total LLM + TTS: {int((t3 - t0)*1000)}ms")

if __name__ == "__main__":
    asyncio.run(benchmark())
