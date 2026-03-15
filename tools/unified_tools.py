from langchain_core.tools import tool
from utils.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
import os

from langchain_community.tools import DuckDuckGoSearchRun
from serpapi import GoogleSearch

@tool
def internet_search(query: str):
    """
    Search the internet for real-time information, current events, news, or general knowledge using DuckDuckGo.
    Use this tool whenever the user asks for information that is not in their documents 
    or requires up-to-date public data.
    """
    try:
        search = DuckDuckGoSearchRun()
        return search.invoke(query)
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
        return "Error: SERPAPI_API_KEY not found in environment. Cannot perform Google Search."
        
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
    
    @tool
    def search_uploaded_documents(query: str):
        """
        Search through the user's uploaded documents (PDFs and Text files).
        Use this tool when the user asks questions about their uploaded files, 
        specific content within documents, or summaries of their files.
        """
        if not vector_store:
            return "Error: No documents have been uploaded or processed yet. Please upload files first."
        
        docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join(
            f"--- Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))} ---\n{doc.page_content}"
            for doc in docs
        )
        return context

    return search_uploaded_documents
