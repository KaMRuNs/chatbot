from langchain_core.tools import tool
import os
import requests

from serpapi import GoogleSearch


def _safe_text(value, default: str) -> str:
    """Convert arbitrary values to readable text without raising formatting errors."""
    if value is None:
        return default
    try:
        text = str(value).strip()
        return text if text else default
    except Exception:
        return default


def _run_duckduckgo_instant_api(query: str, max_results: int = 5) -> str:
    """Fallback DuckDuckGo search via instant answer API when ddgs fails."""
    response = requests.get(
        "https://api.duckduckgo.com/",
        params={
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()

    formatted = []

    answer = _safe_text(payload.get("Answer") or payload.get("AbstractText"), "")
    answer_url = _safe_text(payload.get("AbstractURL"), "")
    if answer:
        line = f"1. Instant Answer\n   Link: {answer_url or 'No URL'}\n   Snippet: {answer}"
        formatted.append(line)

    topics = payload.get("RelatedTopics") or []
    rank = len(formatted) + 1
    for topic in topics:
        if rank > max_results:
            break

        if isinstance(topic, dict) and topic.get("Topics"):
            for nested in topic.get("Topics", []):
                if rank > max_results:
                    break
                title = _safe_text(nested.get("Text"), "No title")
                link = _safe_text(nested.get("FirstURL"), "No URL")
                formatted.append(f"{rank}. {title}\n   Link: {link}\n   Snippet: {title}")
                rank += 1
            continue

        if isinstance(topic, dict):
            title = _safe_text(topic.get("Text"), "No title")
            link = _safe_text(topic.get("FirstURL"), "No URL")
            formatted.append(f"{rank}. {title}\n   Link: {link}\n   Snippet: {title}")
            rank += 1

    if not formatted:
        return "No relevant web results found."

    return "\n\n".join(formatted[:max_results])


def _run_duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Run DuckDuckGo text search with compatibility across ddgs package variants."""
    # Newer implementations use the standalone ddgs package.
    try:
        from ddgs import DDGS  # type: ignore
    except Exception:
        # Older environments may still provide DDGS via duckduckgo_search.
        from duckduckgo_search import DDGS  # type: ignore

    results = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append(item)

    if not results:
        return "No relevant web results found."

    formatted = []
    for index, item in enumerate(results[:max_results], start=1):
        title = _safe_text(item.get("title"), "No title")
        link = _safe_text(item.get("href") or item.get("url"), "No URL")
        snippet = _safe_text(item.get("body") or item.get("snippet"), "No snippet")
        formatted.append(f"{index}. {title}\n   Link: {link}\n   Snippet: {snippet}")

    return "\n\n".join(formatted)

@tool
def internet_search(query: str):
    """
    Search the internet for real-time information, current events, news, or general knowledge using DuckDuckGo.
    Use this tool whenever the user asks for information that is not in their documents 
    or requires up-to-date public data.
    """
    try:
        return _run_duckduckgo_search(query=query, max_results=5)
    except Exception:
        try:
            return _run_duckduckgo_instant_api(query=query, max_results=5)
        except Exception as e:
            return f"Error performing internet search: {e}"

@tool
def google_search_grounding(query: str):
    """
    Perform a highly accurate grounded search using Google Search.
    Use this tool when the user explicitly asks to search Google, or when you need 
    high-confidence factual grounding that DuckDuckGo might miss.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        # Free fallback so the assistant still works without paid APIs.
        return internet_search.invoke(query)
        
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": 3
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "organic_results" in results:
            formatted_results = "\n\n".join([
                f"Title: {r.get('title')}\nLink: {r.get('link')}\nSnippet: {r.get('snippet')}"
                for r in results["organic_results"][:3]
            ])
            return formatted_results
        elif "answer_box" in results:
             return f"Answer: {results['answer_box'].get('answer', results['answer_box'].get('snippet', 'No snippet found'))}"
        else:
            return "No relevant results found on Google."
            
    except Exception as e:
        return f"Error performing Google search: {e}"

def create_retrieval_tool(vector_store):
    """Creates a custom retrieval tool that has access to the provided vector_store."""
    max_chunk_chars = 1200
    
    @tool
    def search_uploaded_documents(query: str):
        """
        Search through the user's uploaded documents (PDFs and Text files).
        Use this tool when the user asks questions about their uploaded files, 
        specific content within documents, or summaries of their files.
        """
        if not vector_store:
            return "Error: No documents have been uploaded or processed yet. Please upload files first."
        
        docs = vector_store.similarity_search(query, k=5)

        # Deduplicate near-identical snippets to avoid wasting context on repeated chunks.
        seen = set()
        formatted_docs = []
        for doc in docs:
            content = (doc.page_content or "").strip()
            if not content:
                continue
            signature = content[:180]
            if signature in seen:
                continue
            seen.add(signature)

            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page")
            page_display = f" | Page {page + 1}" if isinstance(page, int) else ""
            formatted_docs.append(
                f"--- Source: {source}{page_display} ---\n{content[:max_chunk_chars]}"
            )

        if not formatted_docs:
            return "No relevant document context was found for this query."

        context = "\n\n".join(formatted_docs)
        return context

    return search_uploaded_documents
