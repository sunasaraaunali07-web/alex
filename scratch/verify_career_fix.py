import asyncio
import sys
import os

# Configure stdout for utf-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend to sys.path
sys.path.insert(0, os.path.abspath("backend"))

from app.services.llm_service import LLMService

async def main():
    service = LLMService()
    
    career_question = (
        "Can you compare Data Analytics, Machine Learning, and AI Application Development for me? "
        "Which starting path would you recommend for me and why? "
        "Please provide a realistic six-month roadmap, including specific projects to build, "
        "free learning resources, and explain how I can know if I am internship-ready and how to evaluate it."
    )
    
    print("Calling LLMService.generate_response with complex career question...")
    result, duration_ms = await service.generate_response(user_text=career_question)
    
    response_text = result.get("response_text", "")
    spoken_text = result.get("spoken_text", "")
    corrections = result.get("corrections", [])
    
    print(f"\n--- Result in {duration_ms} ms ---")
    print("Corrections count:", len(corrections))
    print("Response text length:", len(response_text))
    print("\n--- Response Text ---")
    print(response_text)
    
    # Verification checks
    assert response_text != "That's interesting! Tell me more.", "FAILED: Defaulted to generic fallback!"
    assert len(response_text) > 300, f"FAILED: Response is too short ({len(response_text)} chars)!"
    
    lower_resp = response_text.lower()
    has_analytics = "analytic" in lower_resp
    has_ml = "machine learning" in lower_resp or "ml" in lower_resp
    has_ai = "ai application" in lower_resp or "ai" in lower_resp
    has_roadmap = "month" in lower_resp or "roadmap" in lower_resp or "phase" in lower_resp
    has_internship = "intern" in lower_resp or "ready" in lower_resp or "evaluate" in lower_resp
    
    print("\n--- Checklist Verification ---")
    print("1. Compares careers:", has_analytics and has_ml and has_ai)
    print("2. Recommendation & reasoning present:", "recommend" in lower_resp or "suggest" in lower_resp or "start" in lower_resp)
    print("3. Roadmap present:", has_roadmap)
    print("4. Projects / resources present:", "project" in lower_resp or "resource" in lower_resp or "course" in lower_resp)
    print("5. Internship readiness present:", has_internship)
    
    assert has_analytics and has_ml, "Missing career comparisons"
    assert has_roadmap, "Missing roadmap"
    assert has_internship, "Missing internship readiness guidance"
    
    print("\n>>> ALL CHECKS PASSED! <<<")

if __name__ == "__main__":
    asyncio.run(main())
