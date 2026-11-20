import asyncio
import time
import os
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))
os.environ["PYTHONPATH"] = str(backend_dir)

from app.services.llm_service import LLMService
from app.services.tts_service import TTSService

async def test_full_pipeline():
    llm = LLMService()
    tts = TTSService()

    text = "Yesterday I was going to college but suddenly my friend called me and said that he is waiting for me since 2 hours. I told him that I already reached there but actually I was still at home. After that he became angry because I didn't inform him earlier and I was not able to explain him what happened. I think he don't understand my situation because whenever I try to explain something he always interrupts me and start arguing with without listening completely."

    print("=== TESTING END-TO-END TURN LATENCY ===")
    t_start = time.perf_counter()
    
    # 1. LLM
    t0 = time.perf_counter()
    llm_res, llm_ms = await llm.generate_response(text)
    t1 = time.perf_counter()
    llm_time = int((t1 - t0) * 1000)
    print(f"1. LLM Generation: {llm_time}ms (reported: {llm_ms}ms)")
    print(f"   Corrections count: {len(llm_res.get('corrections', []))}")

    # 2. TTS
    t2 = time.perf_counter()
    tts_res = await tts.synthesize_turn(
        response_text=llm_res.get("response_text", ""),
        corrections=llm_res.get("corrections", []),
        spoken_text=llm_res.get("spoken_text", "")
    )
    t3 = time.perf_counter()
    tts_time = int((t3 - t2) * 1000)
    print(f"2. TTS Synthesis: {tts_time}ms (reported: {tts_res.get('tts_ms')}ms)")
    print(f"   Segments synthesized: {len(tts_res.get('speech_segments', []))}")

    total_time = int((time.perf_counter() - t_start) * 1000)
    print(f"\nTOTAL TURN TIME: {total_time}ms")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
