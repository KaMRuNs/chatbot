from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm():
    """Returns a ChatOpenAI instance configured for Groq Cloud."""
    return ChatOpenAI(
        model="llama-3.1-8b-instant",  # Switched to a hyper-efficient, open-source 8B model
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        temperature=0.7,
        max_retries=10,
        max_tokens=2048,
    )
