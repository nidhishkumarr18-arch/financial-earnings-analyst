"""Loads API keys and sets up the Gemini LLM."""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or "your_" in api_key:
    print("❌ Set your GOOGLE_API_KEY in the .env file!")
    exit(1)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=api_key,
    temperature=0.1,
)

print("✅ Gemini API configured!")
