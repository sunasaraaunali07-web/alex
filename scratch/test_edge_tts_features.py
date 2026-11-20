import asyncio
import edge_tts

async def test_edge_tts_features():
    voice = "en-US-JennyNeural"
    
    # Test 1: Plain text with capitalization
    print("\n--- Test 1: Capitalization vs Punctuation ---")
    c1 = edge_tts.Communicate("Yesterday, you should say 'I WENT to college' instead of 'I go to college'.", voice)
    d1 = bytearray()
    async for chunk in c1.stream():
        if chunk["type"] == "audio":
            d1.extend(chunk["data"])
    print("Test 1 bytes:", len(d1))

    # Test 2: Does edge_tts accept XML / SSML inside text?
    print("\n--- Test 2: SSML tags in Communicate() ---")
    ssml_text = "Yesterday, you should say <prosody pitch='+5Hz'>I went to college</prosody> instead of I go to college."
    try:
        c2 = edge_tts.Communicate(ssml_text, voice)
        d2 = bytearray()
        async for chunk in c2.stream():
            if chunk["type"] == "audio":
                d2.extend(chunk["data"])
        print("Test 2 bytes:", len(d2))
    except Exception as e:
        print("Test 2 error:", e)

    # Test 3: Segment-based pitch and rate variation
    print("\n--- Test 3: Segment pitch & rate variation ---")
    c3_pre = edge_tts.Communicate("Yesterday, you should say,", voice, rate="-6%", pitch="+0Hz")
    c3_emph = edge_tts.Communicate("I went to college", voice, rate="-12%", pitch="+4Hz")
    c3_post = edge_tts.Communicate("instead of 'I go to college', because you're talking about the past.", voice, rate="-6%", pitch="+0Hz")
    
    d3 = bytearray()
    for comm in [c3_pre, c3_emph, c3_post]:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                d3.extend(chunk["data"])
    print("Test 3 multi-segment audio bytes:", len(d3))

    # Test 4: Can we use quotes or quotation marks in edge-tts?
    print("\n--- Test 4: Quotation marks & commas for natural prosody ---")
    text_prosody = "Yesterday, you should say... 'I went to college'... instead of 'I go to college', because you're talking about the past."
    c4 = edge_tts.Communicate(text_prosody, voice, rate="-6%", pitch="+1Hz")
    d4 = bytearray()
    async for chunk in c4.stream():
        if chunk["type"] == "audio":
            d4.extend(chunk["data"])
    print("Test 4 natural prosody text bytes:", len(d4))

if __name__ == "__main__":
    asyncio.run(test_edge_tts_features())
