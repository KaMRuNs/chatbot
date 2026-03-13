from langchain_core.tools import tool
from utils.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
import os

@tool
def internet_search(query: str):
    """
    Search the internet for real-time information, current events, news, or general knowledge.
    Use this tool whenever the user asks for information that is not in their documents 
    or requires up-to-date data.
    """
    llm = get_llm()
    # Bind the built-in browser_search tool
    llm_with_search = llm.bind(
        tools=[{"type": "browser_search"}],
        tool_choice={"type": "browser_search"},
    )
    
    # We use a simple system prompt for the internal search query
    messages = [
        SystemMessage(content="You are a search tool. Perform the search and return the most relevant info."),
        HumanMessage(content=query)
    ]
    
    response = llm_with_search.invoke(messages)
    return response.content

def create_retrieval_tool(vector_store):
    """Creates a custom retrieval tool that has access to the provided vector_store."""
    
    @tool
    def search_documents(query: str):
        """
        Search through the user's uploaded documents (PDFs and Text files).
        Use this tool when the user asks questions about their uploaded files, 
        specific content within documents, or summaries of their files.
        """
        if not vector_store:
            return "Error: No documents have been uploaded or processed yet. Please upload files first."
        
        docs = vector_store.similarity_search(query, k=5)
        context = "\n\n".join(
            f"--- Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))} ---\n{doc.page_content}"
            for doc in docs
        )
        return context

    return search_documents
