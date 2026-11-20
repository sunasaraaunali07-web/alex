import json
from app.services.llm_service import _parse_llm_json, _validate_response

def test_validate():
    # Case 1: Model returned alternative keys (e.g. 'response', 'answer', 'reply', 'message')
    data1 = {"corrections": [], "response": "Here is the comparison of Data Analytics, ML, and AI..."}
    res1 = _validate_response(data1)
    print("Case 1 (key is 'response'):")
    print("  response_text:", res1["response_text"])

    # Case 2: Model returned section keys
    data2 = {
        "corrections": [],
        "comparison": "Data Analytics vs ML vs AI...",
        "roadmap": "Months 1-2..."
    }
    res2 = _validate_response(data2)
    print("\nCase 2 (keys are section names):")
    print("  response_text:", res2["response_text"])

    # Case 3: Model returned 'response_text' as an empty string
    data3 = {"corrections": [], "response_text": ""}
    res3 = _validate_response(data3)
    print("\nCase 3 ('response_text' is empty):")
    print("  response_text:", res3["response_text"])

    # Case 4: Model returned 'response_text' as a dict or list
    data4 = {"corrections": [], "response_text": {"comparison": "DA vs ML", "roadmap": "6 months"}}
    res4 = _validate_response(data4)
    print("\nCase 4 ('response_text' is dict):")
    print("  response_text:", res4["response_text"])

test_validate()
