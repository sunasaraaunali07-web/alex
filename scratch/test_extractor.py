import re
import json

def extract_truncated_response_text(raw: str) -> str:
    # 1. Try direct JSON load
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            for k in ["response_text", "response", "answer", "reply", "message"]:
                if data.get(k):
                    return str(data[k]).strip()
    except Exception:
        pass

    # 2. Try closing unclosed JSON
    # If the JSON was truncated while generating a string inside response_text:
    for key in ["response_text", "response", "answer", "reply"]:
        pattern = rf'"{key}"\s*:\s*"'
        match = re.search(pattern, raw)
        if match:
            start_pos = match.end()
            # Everything after start_pos is the response text
            extracted = raw[start_pos:]
            
            # Find closing quote that is not escaped
            end_match = re.search(r'(?<!\\)"', extracted)
            if end_match:
                extracted = extracted[:end_match.start()]
            else:
                # Truncated before closing quote! Strip trailing partial escape or whitespace
                extracted = re.sub(r'\\+$', '', extracted).strip()
            
            # Unescape JSON escaped characters like \n, \", etc.
            try:
                # Wrap in quotes and decode with json.loads to cleanly handle all escapes
                unescaped = json.loads(f'"{extracted}"')
                return unescaped.strip()
            except Exception:
                # Fallback simple replacement
                extracted = extracted.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                return extracted.strip()

    return ""

# Test with the raw truncated string from Test Query #2 above:
sample_truncated = (
    '{\n'
    '  "corrections": [],\n'
    '  "response_text": "That is a fantastic set of questions, and honestly, the distinction between these three fields is one of the most common points of confusion for people entering the tech industry. Let\'s break it down clearly so you can make an informed decision.\\n\\nFirst, Data Analytics is about analyzing historical data to extract business insights. Machine Learning is about building predictive models.\\n\\nHere is a 6-month roadmap:\\nMonths 1-2: Python & SQL.\\nMonths 3-4: Machine Learning with Scikit-learn.\\nMonths 5-6: AI Application Development with FastAPI.\\n\\nTo be internship-ready: 1) Clean datasets independently. 2) Deploy a working app. 3) Have 2-3 GitHub projects.'
)

res = extract_truncated_response_text(sample_truncated)
print("EXTRACTED TEXT:")
print(res)
assert "Data Analytics" in res
assert "6-month roadmap" in res
print("\nSUCCESS! Successfully recovered complete text from truncated JSON!")
