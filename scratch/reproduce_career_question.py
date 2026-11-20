import asyncio
import json
import traceback
from app.services.llm_service import LLMService

async def reproduce():
    llm = LLMService()

    career_question = (
        "Can you compare Data Analytics, Machine Learning, and AI Application Development for me? "
        "Which starting path would you recommend for me and why? "
        "Please provide a realistic six-month roadmap, including specific projects to build, "
        "free learning resources, and explain how I can know if I am internship-ready and how to evaluate it."
    )

    print("--- REPRODUCING CAREER GUIDANCE QUESTION ---")
    print("User query:\n", career_question)
    print("\nCalling llm.generate_response()...")

    try:
        res, ms = await llm.generate_response(career_question)
        print(f"\nResponse received in {ms}ms:")
        print("Corrections:", json.dumps(res.get("corrections", []), indent=2))
        print("\nResponse Text:\n", res.get("response_text", ""))
        print("\nSpoken Text:\n", res.get("spoken_text", ""))
        
        if res.get("response_text") == "That's interesting! Tell me more.":
            print("\n>>> REPRODUCED BUG: Alex returned the generic fallback 'That's interesting! Tell me more.' <<<")
        else:
            print("\nDid not return generic fallback. Returned actual answer.")
    except Exception as e:
        print("\nException raised during generate_response:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(reproduce())
