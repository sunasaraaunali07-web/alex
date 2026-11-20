import asyncio
import os
import sys

# Ensure backend is on PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.services.llm_service import (
    LLMService,
    _phrase_exists_in_user_text,
    _overlaps_uncertain_segment,
    _validate_response
)
from app.services.stt_service import STTService


def test_unit_overlaps_uncertain_segment():
    """Unit test the acoustic overlap and containment checker."""
    uncertain_segments = ["and whether was bad", "mumble mumble"]
    
    # Phrase inside uncertain segment
    assert _overlaps_uncertain_segment("whether", uncertain_segments) is True
    assert _overlaps_uncertain_segment("whether was", uncertain_segments) is True
    assert _overlaps_uncertain_segment("whether was bad", uncertain_segments) is True
    assert _overlaps_uncertain_segment("mumble", uncertain_segments) is True

    # Phrase in clear part (NOT in uncertain segments)
    assert _overlaps_uncertain_segment("Yesterday I go", uncertain_segments) is False
    assert _overlaps_uncertain_segment("I go", uncertain_segments) is False
    assert _overlaps_uncertain_segment("to college", uncertain_segments) is False

    # Empty cases
    assert _overlaps_uncertain_segment("", uncertain_segments) is False
    assert _overlaps_uncertain_segment("any word", []) is False
    print("PASS: test_unit_overlaps_uncertain_segment")


def test_unit_validate_response_with_asr():
    """Unit test _validate_response under global and localized ASR conditions."""
    # 1. Global uncertainty suppresses all corrections
    sample_data = {
        "corrections": [
            {"original": "I go", "corrected": "I went", "explanation": "past tense", "type": "grammar"}
        ],
        "response_text": "Sounds great!",
        "spoken_text": "Say 'went'. Sounds great!"
    }
    globally_uncertain_asr = {"is_globally_uncertain": True, "avg_logprob": -1.2}
    res = _validate_response(sample_data, "Yesterday I go to market", globally_uncertain_asr)
    assert res["corrections"] == []
    assert res["spoken_text"] == "Sounds great!"
    print("PASS: test_unit_validate_response_with_asr (global uncertainty)")

    # 2. Localized uncertainty filters out only the uncertain segment error
    sample_multi = {
        "corrections": [
            {"original": "I go", "corrected": "I went", "explanation": "past tense", "type": "grammar"},
            {"original": "whether was", "corrected": "weather was", "explanation": "spelling", "type": "grammar"}
        ],
        "response_text": "Hope it was good!",
        "spoken_text": "You can say 'went'. Hope it was good!"
    }
    localized_asr = {
        "is_globally_uncertain": False,
        "avg_logprob": -0.3,
        "uncertain_segments": ["and whether was very cold"]
    }
    res2 = _validate_response(sample_multi, "Yesterday I go to college and whether was very cold", localized_asr)
    assert len(res2["corrections"]) == 1
    assert res2["corrections"][0]["original"] == "I go"
    assert res2["corrections"][0]["corrected"] == "I went"
    print("PASS: test_unit_validate_response_with_asr (localized filtering)")


async def test_llm_localized_uncertainty_live():
    """Test live LLM behavior when presented with a localized uncertain segment."""
    llm = LLMService()
    user_text = "Yesterday I go to college and whether was very pleasant."
    asr_metadata = {
        "is_globally_uncertain": False,
        "avg_logprob": -0.35,
        "min_logprob": -1.10,
        "uncertain_segments": ["and whether was very pleasant"]
    }
    result, dur = await llm.generate_response(user_text, asr_metadata=asr_metadata)
    print(f"\n[Test Live Localized] Response in {dur}ms:")
    print("Corrections:", result["corrections"])
    print("Spoken:", result["spoken_text"])
    
    # Must correct 'I go' -> 'I went'
    originals = [c["original"].lower() for c in result["corrections"]]
    assert any("go" in orig for orig in originals), f"Expected correction for 'go', got: {result['corrections']}"
    
    # Must NOT correct 'whether' (uncertain acoustic homophone)
    for c in result["corrections"]:
        assert "whether" not in c["original"].lower(), f"Uncertain homophone 'whether' was improperly corrected: {c}"
    print("PASS: test_llm_localized_uncertainty_live")


async def test_llm_global_uncertainty_live():
    """Test live LLM behavior when entire turn is globally uncertain."""
    llm = LLMService()
    user_text = "Yesterday I go to college and he don't want it."
    asr_metadata = {
        "is_globally_uncertain": True,
        "avg_logprob": -1.15,
        "no_speech_prob": 0.55
    }
    result, dur = await llm.generate_response(user_text, asr_metadata=asr_metadata)
    print(f"\n[Test Live Global Uncertain] Response in {dur}ms:")
    print("Corrections:", result["corrections"])
    print("Spoken:", result["spoken_text"])
    
    assert result["corrections"] == [], f"Expected 0 corrections for globally uncertain turn, got {result['corrections']}"
    assert result["spoken_text"] == result["response_text"], "Spoken text must equal conversational response text"
    print("PASS: test_llm_global_uncertainty_live")


async def test_real_stt_metadata_extraction():
    """Test real Whisper STT to verify new asr_metadata dictionary structure."""
    import edge_tts
    import io

    communicate = edge_tts.Communicate("Hello Alex, I am testing the new acoustic confidence pipeline.", "en-US-JennyNeural")
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]

    stt = STTService()
    transcript, dur, asr_meta = await stt.transcribe(audio_bytes, "test.mp3")
    print(f"\n[Real STT Check] Transcribed: '{transcript}' in {dur}ms")
    print("ASR Metadata Keys:", list(asr_meta.keys()))
    print(f"avg_logprob: {asr_meta.get('avg_logprob')}")
    print(f"min_logprob: {asr_meta.get('min_logprob')}")
    print(f"no_speech_prob: {asr_meta.get('no_speech_prob')}")
    print(f"has_segments: {asr_meta.get('has_segments')}")
    print(f"uncertain_segments count: {len(asr_meta.get('uncertain_segments', []))}")
    print(f"is_globally_uncertain: {asr_meta.get('is_globally_uncertain')}")

    assert asr_meta["has_segments"] is True
    assert asr_meta["avg_logprob"] is not None
    assert asr_meta["min_logprob"] is not None
    assert isinstance(asr_meta["segments"], list)
    assert isinstance(asr_meta["uncertain_segments"], list)
    assert isinstance(asr_meta["is_globally_uncertain"], bool)
    print("PASS: test_real_stt_metadata_extraction")


async def main():
    print("=== Running Step 3 ASR Uncertainty Detection Tests ===")
    test_unit_overlaps_uncertain_segment()
    test_unit_validate_response_with_asr()
    await test_llm_localized_uncertainty_live()
    await test_llm_global_uncertainty_live()
    await test_real_stt_metadata_extraction()
    print("\nALL STEP 3 TESTS PASSED PERFECTLY!")


if __name__ == "__main__":
    asyncio.run(main())
