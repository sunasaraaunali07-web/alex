import asyncio
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("backend"))

from app.services.llm_service import LLMService

async def main():
    service = LLMService()
    
    test_turn = "Yesterday I go to college and my professor tell me about data analytics. What kind of job I can get with data analytics?"
    
    print("Testing turn with both grammar mistakes AND a real career question...")
    result, duration_ms = await service.generate_response(user_text=test_turn)
    
    corrections = result.get("corrections", [])
    response_text = result.get("response_text", "")
    spoken_text = result.get("spoken_text", "")
    
    print(f"\n--- Result in {duration_ms} ms ---")
    print("Corrections found:", len(corrections))
    for i, c in enumerate(corrections, 1):
        print(f"Correction #{i}: '{c['original']}' -> '{c['corrected']}'")
        print(f"Spoken comparison: {c.get('spoken_comparison')}")
        
    print("\nResponse Text:\n", response_text)
    
    assert len(corrections) >= 1, "Expected at least 1 grammar correction"
    assert "That's interesting! Tell me more." not in response_text, "Should not return fallback"
    assert "analyst" in response_text.lower() or "job" in response_text.lower() or "role" in response_text.lower(), "Should answer the job question"
    
    print("\n>>> DUAL-ROLE TEST PASSED! <<<")

if __name__ == "__main__":
    asyncio.run(main())
