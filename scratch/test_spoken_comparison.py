import asyncio
import edge_tts

async def test_comparison_prosody():
    voice = "en-US-JennyNeural"
    
    # Let's test two approaches:
    # Approach A: Single unified sentence with natural prosodic punctuation (ellipses & commas)
    unified_text = (
        "In that first sentence, you should say... 'I went to college'... "
        "instead of 'I go to college'—because you're talking about the past."
    )
    c_unified = edge_tts.Communicate(unified_text, voice, rate="-6%", pitch="+0Hz")
    d_unified = bytearray()
    async for chunk in c_unified.stream():
        if chunk["type"] == "audio":
            d_unified.extend(chunk["data"])
    print("Approach A (Unified prosodic text) audio bytes:", len(d_unified))

    # Approach B: Segmented with distinct pitch/rate for the corrected phrase
    # Segment 1: Lead-in
    c1 = edge_tts.Communicate("In that first sentence, you should say,", voice, rate="-6%", pitch="+0Hz")
    # Segment 2: Emphasized corrected phrase (clear, slightly slower rate -10%, pitch +2Hz)
    c2 = edge_tts.Communicate("'I went to college'", voice, rate="-10%", pitch="+2Hz")
    # Segment 3: Contrast and brief reason
    c3 = edge_tts.Communicate("instead of 'I go to college', because you're talking about the past.", voice, rate="-6%", pitch="+0Hz")

    d_segmented = bytearray()
    for c in [c1, c2, c3]:
        async for chunk in c.stream():
            if chunk["type"] == "audio":
                d_segmented.extend(chunk["data"])
    print("Approach B (Segmented prosody with pitch/rate boost) audio bytes:", len(d_segmented))

if __name__ == "__main__":
    asyncio.run(test_comparison_prosody())
