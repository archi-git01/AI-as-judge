# AI-As-A-Judge: LLM Chatbot Evaluation Framework

An advanced, independent QA Automation framework leveraging Google Gemini 3.8 Flash and Pytest to perform non-deterministic validation on Generative AI applications.

## 🚀 Key Features
- **AI-Driven Assertions**: Bypasses rigid hardcoded text assertions by utilizing an LLM acting as a QA Quality Check.
- **Structured System Prompts**: Standardizes unstructured chatbot outputs into reliable, machine-readable JSON payloads.
- **Platform Isolation**: Built completely within a controlled Python virtual environment.

## 🛠️ Tech Stack
- **Language**: Python 3.14+
- **Test Engine**: Pytest
- **AI Evaluator**: Google GenAI SDK (Gemini 3.8 Flash)

## 🏃 Run the Project Locally
1. Clone this repository.
2. Run `pip install -r requirements.txt`.
3. Set your environment variable: `export GEMINI_API_KEY="your_key"`.
4. Execute: `pytest -s -v tests/test_chatbot.py`.
