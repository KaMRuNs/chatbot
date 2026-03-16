from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

DEFAULT_PRIMARY_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_SECONDARY_MODEL = "llama-3.1-8b-instant"


def get_model_candidates() -> list[str]:
    """Returns ordered model candidates for automatic failover."""
    primary = os.getenv("GROQ_MODEL") or os.getenv("GROQ_MODEL_PRIMARY") or DEFAULT_PRIMARY_MODEL
    secondary = os.getenv("GROQ_MODEL_FALLBACK") or DEFAULT_SECONDARY_MODEL

    candidates = [primary]
    if secondary != primary:
        candidates.append(secondary)
    return candidates


def is_token_limit_error(error: Exception) -> bool:
    """Checks whether an exception looks like a token/context-length failure."""
    message = str(error).lower()
    markers = (
        "context length",
        "context_length_exceeded",
        "maximum context",
        "too many tokens",
        "token limit",
        "prompt is too long",
        "request too large",
    )
    return any(marker in message for marker in markers)


def get_llm(model_name: str | None = None, document_mode: bool = False):
    """Returns a ChatOpenAI instance configured for Groq Cloud."""
    resolved_model = model_name or get_model_candidates()[0]
    temperature = 0.3 if document_mode else 0.7
    max_tokens = 640 if document_mode else 768
    return ChatOpenAI(
        model=resolved_model,
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        temperature=temperature,
        max_retries=10,
        # Keep answers concise to avoid context-window pressure on smaller models.
        max_tokens=max_tokens,
    )
