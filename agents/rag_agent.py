from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.llm import get_llm, get_model_candidates, is_token_limit_error
from tools.ocr_tool import pytesseract
import tempfile
import re
import os
import io
import shutil

import pypdf
from PIL import Image, ImageOps, ImageEnhance


def sanitize_text(text: str) -> str:
    """Remove unsafe control characters while preserving document structure."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Keep paragraph and list structure while trimming noisy trailing spaces.
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_text(text: str) -> str:
    """Alias for sanitize_text for backward compatibility."""
    return sanitize_text(text)


def _resolve_tesseract_cmd() -> str | None:
    """Resolve Tesseract executable path if available; return None when unavailable."""
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and os.path.exists(env_cmd):
        return env_cmd

    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path

    found = shutil.which("tesseract")
    return found


def _preprocess_pil_for_ocr(image: Image.Image) -> Image.Image:
    """Normalize image for better OCR quality."""
    img = ImageOps.exif_transpose(image)
    gray = ImageOps.grayscale(img)
    return ImageEnhance.Contrast(gray).enhance(1.6)


def _ocr_page_images(page) -> str:
    """Run OCR over extracted images in a PDF page and return merged text."""
    image_text_parts = []
    for image_file in getattr(page, "images", []):
        try:
            img = Image.open(io.BytesIO(image_file.data))
            prepared = _preprocess_pil_for_ocr(img)
            text = pytesseract.image_to_string(prepared)
            text = sanitize_text(text)
            if text:
                image_text_parts.append(text)
        except Exception:
            continue
    return "\n\n".join(image_text_parts).strip()


def _load_pdf_with_ocr_fallback(pdf_path: str, source_name: str) -> list[Document]:
    """Load PDF per page and use OCR fallback for image-based pages."""
    reader = pypdf.PdfReader(pdf_path)
    docs: list[Document] = []

    tesseract_cmd = _resolve_tesseract_cmd()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    for page_index, page in enumerate(reader.pages):
        direct_text = sanitize_text(page.extract_text() or "")
        page_text = direct_text

        # Scanned PDFs often have little/no extractable text. Fallback to page-image OCR.
        if len(page_text) < 40 and tesseract_cmd:
            ocr_text = _ocr_page_images(page)
            if ocr_text:
                if page_text:
                    page_text = f"{page_text}\n\n{ocr_text}".strip()
                else:
                    page_text = ocr_text

        if page_text:
            docs.append(
                Document(
                    page_content=page_text,
                    metadata={"source": source_name, "page": page_index},
                )
            )

    return docs


def build_vector_store(uploaded_files):
    """Loads uploaded PDF/TXT files, splits into chunks, and creates a FAISS vector store."""
    all_docs = []
    failed_files = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            # Load based on file type
            if suffix == ".pdf":
                docs = _load_pdf_with_ocr_fallback(tmp_path, uploaded_file.name)
            elif suffix == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
                docs = loader.load()
            else:
                continue

            for doc in docs:
                doc.page_content = sanitize_text(doc.page_content)
                if "source" not in doc.metadata or not doc.metadata.get("source"):
                    doc.metadata["source"] = uploaded_file.name
            if docs:
                all_docs.extend(docs)
            else:
                failed_files.append(f"{uploaded_file.name} (empty text)")
        except Exception as e:
            failed_files.append(f"{uploaded_file.name}: {str(e)}")
        finally:
            os.unlink(tmp_path)

    if not all_docs:
        details = "; ".join(failed_files) if failed_files else "No readable text found."
        raise ValueError(f"Unable to process uploaded files. {details}")

    # Split documents into small chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=220,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


def ask_stream(question: str, vector_store, chat_history: list = None):
    """Streams the RAG response and returns retrieved chunks for display."""
    # Step 1: Retrieve relevant chunks
    results = vector_store.similarity_search_with_score(question, k=5)
    filtered = [item for item in results if item[1] <= 1.9]
    docs = [doc for doc, _ in (filtered if filtered else results)]

    context_parts = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page")
        page_display = f"Page {page + 1}" if isinstance(page, int) else "Page N/A"
        context_parts.append(
            f"[Source: {source} | {page_display}]\n{sanitize_text(doc.page_content)}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Step 2: Build messages with chat history
    system_prompt = (
        "You are a precise document analyst. Use only the provided document context.\n\n"
        "Response format:\n"
        "1) Direct Answer: 1-3 lines.\n"
        "2) Key Evidence: short bullet list quoting or paraphrasing from the context.\n"
        "3) Sources: bullet list in the form '- filename (page X)'.\n\n"
        f"Context:\n{context}\n\n"
        "If evidence is weak or missing, state that clearly and ask one focused follow-up question."
    )

    messages = [SystemMessage(content=system_prompt)]

    # Add chat history for follow-up questions
    if chat_history:
        for msg in chat_history[-4:]:  # Last 2 exchanges to save tokens
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    # Step 3: Stream the response with model failover on token-limit errors
    last_error = None
    for model_name in get_model_candidates():
        llm = get_llm(model_name=model_name, document_mode=True)
        try:
            for chunk in llm.stream(messages):
                if chunk.content:
                    yield chunk.content
            return
        except Exception as e:
            last_error = e
            if is_token_limit_error(e):
                continue
            raise

    if last_error:
        raise last_error


def get_retrieved_chunks(question: str, vector_store) -> list:
    """Returns the retrieved chunks with similarity scores for display."""
    # similarity_search_with_score returns (doc, score) pairs
    # FAISS returns L2 distance — lower = more similar
    results = vector_store.similarity_search_with_score(question, k=5)

    chunks = []
    seen = set()
    for doc, score in results:
        # Convert L2 distance to a 0-100% relevance score
        relevance = max(0, round((1 / (1 + score)) * 100, 1))
        if relevance < 20:
            continue

        content = clean_text(doc.page_content)
        signature = content[:180]
        if signature in seen:
            continue
        seen.add(signature)

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", None)
        chunks.append({
            "content": content,
            "source": os.path.basename(source) if source != "Unknown" else source,
            "page": page,
            "score": score,
            "relevance": relevance,
        })

    return chunks
