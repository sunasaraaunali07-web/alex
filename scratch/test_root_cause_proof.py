import json
import re
from app.services.llm_service import _parse_llm_json, _validate_response, _fallback_response

def test_truncation_recovery():
    print("--- TESTING TRUNCATED JSON RECOVERY ---")
    
    # Simulate a truncated JSON response where max_tokens cut off the text before closing quote and brace:
    truncated_raw = (
        '{\n'
        '  "corrections": [],\n'
        '  "response_text": "Data Analytics is about interpreting historical data, while Machine Learning builds predictive models. '
        'For beginners, I recommend Data Analytics. Here is a 6-month roadmap: Months 1-2 learn Python and SQL... '
        'Months 3-4 learn Scikit-learn and build prediction models... '
        'Months 5-6 build AI applications with FastAPI... '
        'To be internship ready, build 2-3 portfolio projects on GitHub and practice SQL.'
        # Notice: no closing quote, no closing brace!
    )

    print("Simulated raw output from truncated model:")
    print(truncated_raw[:150] + "...")

    # Current parser output:
    curr_res = _parse_llm_json(truncated_raw)
    print("\nCURRENT PARSER RESULT:")
    print("response_text:", repr(curr_res["response_text"]))
    
    # Check if it was wiped out by the fallback:
    if curr_res["response_text"] in ["I see! Tell me more about that.", "That's interesting! Tell me more."]:
        print(">>> CONFIRMED: Current parser completely wiped out the 400-word answer and replaced it with generic fallback! <<<")

if __name__ == "__main__":
    test_truncation_recovery()
