import asyncio
import edge_tts
import os

async def generate_audio_samples():
    voice = "en-US-JennyNeural"
    out_dir = os.path.join(os.path.dirname(__file__), "audio_samples")
    os.makedirs(out_dir, exist_ok=True)

    # Style 1: Prosodically punctuated single segment per correction
    # Using ellipses (...) and quotes for natural breathing pauses and emphasis
    s1_text = "In your sentence, you should say... 'I went to college'... instead of 'I go to college'—because you're talking about the past."
    c1 = edge_tts.Communicate(s1_text, voice, rate="-6%", pitch="+1Hz")
    await c1.save(os.path.join(out_dir, "style1_prosodic_punctuation.mp3"))

    # Style 2: Segmented sub-units with pitch/rate shift on the corrected phrase
    c2_lead = edge_tts.Communicate("In your sentence, you should say...", voice, rate="-6%", pitch="+0Hz")
    c2_emph = edge_tts.Communicate("'I went to college'", voice, rate="-10%", pitch="+3Hz")
    c2_rest = edge_tts.Communicate("instead of 'I go to college'—because you're talking about the past.", voice, rate="-6%", pitch="+0Hz")

    d2 = bytearray()
    for c in [c2_lead, c2_emph, c2_rest]:
        async for chunk in c.stream():
            if chunk["type"] == "audio":
                d2.extend(chunk["data"])
    with open(os.path.join(out_dir, "style2_subsegmented.mp3"), "wb") as f:
        f.write(d2)

    print("Generated audio samples in", out_dir)
    print("Style 1 size:", os.path.getsize(os.path.join(out_dir, "style1_prosodic_punctuation.mp3")))
    print("Style 2 size:", os.path.getsize(os.path.join(out_dir, "style2_subsegmented.mp3")))

if __name__ == "__main__":
    asyncio.run(generate_audio_samples())
