import os
import json
import pytest
from google import genai
from utils.ai_judge import evaluate_response

def load_test_scenarios():
    """Reads our localized JSON intent data file dynamically."""
    with open("tests/test_data.json", "r") as file:
        return json.load(file)

def get_live_chatbot_response(user_prompt):
    """Generates a live, real-time response from the Chatbot system under test."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Error: GEMINI_API_KEY environment variable is missing.")
        
    client = genai.Client(api_key=api_key)
    
    # Simulates hitting your company's live chatbot backend application at runtime
    response = client.models.generate_content(
        model='gemini-3.8-flash',
        contents=user_prompt
    )
    return response.text

# Parametrize the test runner to generate unique test iterations per prompt
@pytest.mark.parametrize("case", load_test_scenarios())
def test_live_data_driven_pipeline(case):
    """Executes a dynamic end-to-end loop: Read Prompt -> Call Bot -> AI Judge Evaluation."""
    print(f"\n\n🎬 Running Scenario: [{case['scenario']}]")
    print(f"👉 User Prompt sent to Bot: '{case['user_prompt']}'")
    
    # 1. Fetch live production response
    live_chatbot_reply = get_live_chatbot_response(case['user_prompt'])
    print(f"🤖 Chatbot Live Generated Response:\n---\n{live_chatbot_reply}\n---")
    
    # 2. Grade live response using distinct AI Judge utility
    print("⚖️ Submitting live transaction variables to AI Judge...")
    verdict = evaluate_response(case['user_prompt'], live_chatbot_reply)
    
    print(f"Verdict Outcome: {verdict.get('status')}")
    print(f"Quality Metric: {verdict.get('accuracy_score')}/5")
    print(f"Judge Analysis: {verdict.get('reason')}")
    
    # 3. Fail the automation run if the live chatbot generates an invalid or unsafe response
    assert verdict["status"] == "PASS", f"Test failed due to: {verdict.get('reason')}"