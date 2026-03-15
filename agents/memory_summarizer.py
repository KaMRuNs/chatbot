"""
Memory Summarizer — Compresses long chat histories into a concise running summary.
Called from app.py when message count exceeds the configured threshold.
"""

from utils.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

SUMMARIZE_THRESHOLD = 12  # Summarize when chat exceeds this many messages
KEEP_RECENT = 6           # Always keep these many recent messages intact


def should_summarize(messages: list) -> bool:
    """Returns True if the message list is long enough to warrant summarization."""
    return len(messages) > SUMMARIZE_THRESHOLD


def summarize_messages(messages: list) -> str:
    """
    Compresses a list of chat messages into a concise narrative summary.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys.
    
    Returns:
        A concise text summary of the conversation.
    """
    llm = get_llm()

    # Format messages into a readable transcript
    transcript = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:500]  # Truncate very long messages
        transcript += f"{role}: {content}\n\n"

    system_prompt = (
        "You are a conversation summarizer. "
        "Given a chat transcript, produce a concise summary (max 200 words) that captures:\n"
        "- What the user asked about\n"
        "- What actions were taken (emails sent, jobs searched, etc.)\n"
        "- Key decisions or information shared\n"
        "Write in third person, past tense. Be factual, not interpretive."
    )

    user_prompt = f"Summarize this conversation:\n\n{transcript}"

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        return response.content.strip()
    except Exception as e:
        return f"[Summary unavailable: {e}]"


def compress_session(session: dict) -> dict:
    """
    Takes a full session dict and returns a compressed version:
    - Old messages are summarized and stored in session["summary"]
    - Only the last KEEP_RECENT messages are kept in session["messages"]
    
    Args:
        session: A session dict with "messages" and optionally "summary".
    
    Returns:
        The updated session dict.
    """
    messages = session.get("messages", [])
    if not should_summarize(messages):
        return session

    to_summarize = messages[:-KEEP_RECENT]
    to_keep = messages[-KEEP_RECENT:]

    existing_summary = session.get("summary", "")
    new_summary_text = summarize_messages(to_summarize)

    # Prepend existing summary if it exists
    if existing_summary:
        combined = f"{existing_summary}\n\nMore recently: {new_summary_text}"
    else:
        combined = new_summary_text

    session["summary"] = combined
    session["messages"] = to_keep
    return session
