from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.llm import get_llm
import tempfile
import re
import os


def sanitize_text(text: str) -> str:
    """Removes all non-printable characters and null bytes to prevent JSON parsing errors."""
    if not text:
        return ""
    # Remove null bytes and other common problematic control characters
    text = text.replace("\x00", "").replace("\x0b", "").replace("\x0c", "")
    # Keep only printable characters (ASCII range 32-126 + common whitespace like newlines/tabs)
    # This prevents emojis and non-standard symbols from breaking older JSON parsers if needed,
    # though modern ones handle them, Groq/Llama can be picky with huge strings.
    # More importantly, it collapses weird whitespace that confuses some regex.
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    text = re.sub(r'\s+', ' ', text)  # Collapse all whitespace to single spaces
    return text.strip()

def clean_text(text: str) -> str:
    """Alias for sanitize_text for backward compatibility."""
    return sanitize_text(text)


def build_vector_store(uploaded_files):
    """Loads uploaded PDF/TXT files, splits into chunks, and creates a FAISS vector store."""
    all_docs = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        # Load based on file type
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".txt":
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            os.unlink(tmp_path)
            continue

        docs = loader.load()
        all_docs.extend(docs)
        os.unlink(tmp_path)

    # Split documents into small chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


def ask_stream(question: str, vector_store, chat_history: list = None):
    """Streams the RAG response and returns retrieved chunks for display."""
    llm = get_llm()

    # Step 1: Retrieve relevant chunks
    docs = vector_store.similarity_search(question, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Step 2: Build messages with chat history
    system_prompt = (
        "Be concise. Answer in short paragraphs. Use bullets only when listing items.\n\n"
        f"Context:\n{context}\n\n"
        "If the answer is not in the context, say you don't know."
    )

    messages = [SystemMessage(content=system_prompt)]

    # Add chat history for follow-up questions
    if chat_history:
        for msg in chat_history[-6:]:  # Last 3 exchanges to save tokens
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    # Step 3: Stream the response
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def get_retrieved_chunks(question: str, vector_store) -> list:
    """Returns the retrieved chunks with similarity scores for display."""
    # similarity_search_with_score returns (doc, score) pairs
    # FAISS returns L2 distance — lower = more similar
    results = vector_store.similarity_search_with_score(question, k=5)

    chunks = []
    for doc, score in results:
        # Convert L2 distance to a 0-100% relevance score
        relevance = max(0, round((1 / (1 + score)) * 100, 1))
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", None)
        chunks.append({
            "content": clean_text(doc.page_content),
            "source": os.path.basename(source) if source != "Unknown" else source,
            "page": page,
            "score": score,
            "relevance": relevance,
        })

    return chunks
