from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm():
    """Returns a ChatOpenAI instance configured for Groq Cloud."""
    return ChatOpenAI(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        temperature=0.7,
        max_tokens=1024,  # Cap output tokens to save quota on free tier
    )
