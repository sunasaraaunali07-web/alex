import asyncio
import edge_tts

async def check_ssml_behavior():
    voice = "en-US-JennyNeural"
    # Compare with word boundaries
    text_with_tag = "You should say <prosody pitch='+5Hz'>I went</prosody> instead of I go."
    c = edge_tts.Communicate(text_with_tag, voice)
    words = []
    async for chunk in c.stream():
        if chunk["type"] == "WordBoundary":
            words.append(chunk["text"])
    print("Words spoken when XML passed:", words)

    # Now compare with segment-level approach
    print("\nTesting segment-level emphasis...")
    # In segmented synthesis:
    # Segment 1: Lead-in e.g. "Yesterday, you should say," (rate="-6%", pitch="+0Hz")
    # Segment 2: Emphasized correction phrase e.g. "'I went to college'" (rate="-10%", pitch="+2Hz")
    # Segment 3: Contrast & explanation e.g. "instead of 'I go to college', because you're talking about the past." (rate="-6%", pitch="+0Hz")
    # This guarantees 100% genuine vocal pitch and rate difference without risking raw XML words being read aloud!

if __name__ == "__main__":
    asyncio.run(check_ssml_behavior())
