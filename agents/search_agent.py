import re
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.llm import get_llm

SYSTEM_PROMPT = (
    "You are a search assistant. You MUST use the browser_search tool for EVERY user query — "
    "no exceptions. Never answer from memory alone; always search the internet first, "
    "then use the search results to form your response.\n"
    "Be concise. Answer in short paragraphs. Use bullets only when listing items.\n"
    "No tables. No citation markers like 【1†L5】.\n"
    "End with: Sources:\n- [Title](URL)"
)


def clean_chunk(text: str) -> str:
    """Cleans up any remaining citation markers and HTML tags from a chunk."""
    text = re.sub(r"【.*?】", "", text)
    text = text.replace("<br>", "\n")
    return text


def search_stream(query: str, chat_history: list = None):
    """Streams the search response token by token with chat memory."""
    llm = get_llm()

    llm_with_search = llm.bind(
        tools=[{"type": "browser_search"}],
        tool_choice={"type": "browser_search"},
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Add chat history for follow-up questions
    if chat_history:
        for msg in chat_history[-6:]:  # Last 3 exchanges to save tokens
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=query))

    try:
        for chunk in llm_with_search.stream(messages):
            if chunk.content:
                yield clean_chunk(chunk.content)
    except Exception as e:
        if "tool" in str(e).lower():
            # Fallback: retry without tool_choice constraint
            llm_fallback = llm.bind(tools=[{"type": "browser_search"}])
            for chunk in llm_fallback.stream(messages):
                if chunk.content:
                    yield clean_chunk(chunk.content)
        else:
            raise
