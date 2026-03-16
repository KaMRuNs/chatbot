import streamlit as st
import json
import os
import uuid
import re
import base64
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables at the very beginning
load_dotenv(override=True)

ACTIVE_GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

from agents.rag_agent import build_vector_store, sanitize_text
from agents.action_agent import action_agent_stream
from agents.memory_summarizer import compress_session, should_summarize
from tools.email_tool import set_email_context
import pypdf
import io
import tempfile
st.set_page_config(
    page_title="CareerPilot", 
    page_icon="🤖", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---- PDF Export Helper ----
def generate_pdf_bytes(messages: list, title: str = "Chat Export") -> bytes:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 10)
    for msg in messages:
        role = "You" if msg["role"] == "user" else "Assistant"
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, f"{role}:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        safe_content = msg["content"].encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 5, safe_content)
        pdf.ln(3)
    return pdf.output(dest='S').encode('latin-1')

# ---- Custom CSS ----
st.markdown("""
<style>
    /* ChatGPT-inspired Dark Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #212121;
    }

    .stApp {
        background-color: #212121;
    }

    /* Minimalist SideBar */
    section[data-testid="stSidebar"] {
        background-color: #171717 !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Modern Typography */
    h1 {
        color: #ececec;
        font-weight: 600;
        letter-spacing: -0.5px;
        margin-bottom: 0px !important;
        text-align: center;
    }

    .chatgpt-subheader {
        font-size: 1rem;
        color: #b4b4b4;
        margin-bottom: 2rem;
        text-align: center;
    }

    /* Chat Message Bubbles */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 12px 0 !important;
        max-width: 800px;
        margin: 0 auto;
    }

    /* Hide Streamlit Default Avatars */
    .stChatMessage > div:first-child,
    div[data-testid="stChatMessageAvatar"] img {
        display: none !important;
    }

    /* User Message Style */
    div[data-testid="stChatMessage"]:has(img[src*="user-avatar"]) > div:last-child {
        background-color: #2f2f2f !important;
        border-radius: 20px !important;
        padding: 12px 20px !important;
        margin-left: auto;
        width: fit-content;
        max-width: 75%;
        color: #ececec;
        font-size: 1rem;
        line-height: 1.5;
    }

    /* Assistant Message Style */
    div[data-testid="stChatMessage"]:has(img[src*="assistant-avatar"]) > div:last-child {
        padding-left: 0 !important;
        color: #ececec;
        font-size: 1rem;
        line-height: 1.6;
        width: 100%;
    }

    /* Chat Input Fixed to Bottom */
    .stChatInputContainer {
        border-radius: 20px !important;
        background-color: #2f2f2f !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        padding: 5px 15px !important;
        transition: border-color 0.3s ease;
    }

    .stChatInputContainer:focus-within {
        border-color: rgba(255,255,255,0.3) !important;
    }

    /* Hide redundant elements */
    /* Hide redundant elements (restoring header for sidebar toggle) */
    #MainMenu, footer {visibility: hidden;}
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #212121;
    }
    ::-webkit-scrollbar-thumb {
        background: #424242;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #565656;
    }

    /* Buttons */
    div[data-testid="stButton"] button {
        background-color: #2f2f2f;
        border: 1px solid rgba(255,255,255,0.1);
        color: #ececec;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
    }

    div[data-testid="stButton"] button:hover {
        background-color: #424242;
        border-color: rgba(255,255,255,0.2);
    }

</style>
""", unsafe_allow_html=True)

st.title("CareerPilot")
st.markdown('<div class="chatgpt-subheader">Search, Reason, and Act.</div>', unsafe_allow_html=True)

SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def normalize_response_format(text: str) -> str:
    """Clean noisy spacing and bullet formatting while preserving content."""
    if not text:
        return text
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\n\s*[-*]\s*", "\n- ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def render_chunks(chunks):
    """Renders retrieved chunks with relevance scores and metadata."""
    for i, chunk in enumerate(chunks, 1):
        relevance = chunk.get("relevance", 0)
        score = chunk.get("score", 0)
        source = chunk.get("source", "Unknown")
        page = chunk.get("page")

        # Color based on relevance
        if relevance >= 60:
            color_class = "relevance-high"
        elif relevance >= 40:
            color_class = "relevance-mid"
        else:
            color_class = "relevance-low"

        # Header with chunk number, relevance, and metadata
        page_info = f" | Page {page + 1}" if page is not None else ""
        st.markdown(
            f'<div class="chunk-header">'
            f'<strong>Chunk {i}</strong>'
            f'<span class="chunk-meta">'
            f'<span class="{color_class}">Relevance: {relevance}%</span>'
            f' | Distance: {score:.3f}'
            f'{page_info}'
            f' | Source: {source}'
            f'</span></div>',
            unsafe_allow_html=True,
        )
        st.code(chunk["content"], language=None)
        if i < len(chunks):
            st.divider()


# ---- Sidebar ----
with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_id] = {"title": "New Chat", "messages": [], "updated_at": datetime.now().isoformat()}
        st.session_state.current_session_id = new_id
        st.rerun()

    st.header("💬 Chat History")
    # sort by updated_at descending
    if "chat_sessions" in st.session_state:
        sorted_sessions = sorted(st.session_state.chat_sessions.items(), key=lambda x: x[1]['updated_at'], reverse=True)
        for s_id, s_data in sorted_sessions:
            title = s_data["title"]
            # Highlight active session
            btn_label = f"🟢 {title}" if s_id == st.session_state.current_session_id else f"⚪ {title}"
            if st.button(btn_label, key=f"btn_{s_id}", use_container_width=True):
                st.session_state.current_session_id = s_id
                st.rerun()
    
    st.divider()
    
    st.header("Assistant Control")
    st.divider()

    # Resume Context Control
    st.subheader("📄 Resume Profile")
    if "resume_text" not in st.session_state:
        st.markdown("<small><i>Upload your resume in the chat to enable job matching tools.</i></small>", unsafe_allow_html=True)
    else:
        st.success("Resume loaded and ready for action.")
        if st.button("Clear Resume", use_container_width=True):
             del st.session_state["resume_text"]
             st.rerun()

    # Status & Legend
    st.markdown("---")
    if "vector_store" in st.session_state:
        st.caption("✅ Knowledge Base Active")
    else:
        st.caption("💡 Knowledge Base Empty")
    
    if "resume_text" in st.session_state:
        st.caption("✅ Resume Background Active")

    st.caption("🛠️ Web and Email are always ON")

    st.divider()

    # Clear chat button
    if st.button("Clear Current Chat", use_container_width=True):
        if "current_session_id" in st.session_state:
            st.session_state.chat_sessions[st.session_state.current_session_id]["messages"] = []
        st.rerun()

    # PDF Export button
    if "current_session_id" in st.session_state:
        current_messages = st.session_state.chat_sessions[st.session_state.current_session_id]["messages"]
        if current_messages:
            try:
                pdf_bytes = generate_pdf_bytes(
                    current_messages,
                    title=st.session_state.chat_sessions[st.session_state.current_session_id].get("title", "Chat Export")
                )
                st.download_button(
                    "📄 Download Chat as PDF",
                    data=pdf_bytes,
                    file_name="careerpilot_chat.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception:
                # fpdf fallback
                chat_export = ""
                for msg in current_messages:
                    role = "You" if msg["role"] == "user" else "Assistant"
                    chat_export += f"{role}:\n{msg['content']}\n\n---\n\n"
                st.download_button(
                    "💾 Export Chat",
                    data=chat_export,
                    file_name="careerpilot_chat.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    # Move technical info to bottom
    st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sidebar-info">Engine: Groq Cloud (Open-Source)<br>Model: {ACTIVE_GROQ_MODEL}</p>',
        unsafe_allow_html=True
    )

# ---- Initialize Chat History State ----
if "chat_sessions" not in st.session_state:
    default_id = str(uuid.uuid4())
    st.session_state.chat_sessions = {
        default_id: {"title": "New Chat", "messages": [], "updated_at": datetime.now().isoformat()}
    }
    st.session_state.current_session_id = default_id

current_id = st.session_state.current_session_id
messages = st.session_state.chat_sessions[current_id]["messages"]

# ---- Custom Avatars for CSS Targeting ----
USER_AVATAR = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' data-id='user-avatar'/%3E"
ASSISTANT_AVATAR = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' data-id='assistant-avatar'/%3E"

# ---- Display Chat History ----
for i, message in enumerate(messages):
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        # Check if response contains a calendar event file path and show download button
        if message["role"] == "assistant":
            ics_match = re.search(r'ICS_FILE_PATH:(.+\.ics)', message["content"])
            if ics_match:
                ics_path = ics_match.group(1).strip()
                if os.path.exists(ics_path):
                    with open(ics_path, "rb") as f:
                        st.download_button(
                            "📅 Add to Calendar",
                            data=f.read(),
                            file_name=os.path.basename(ics_path),
                            mime="text/calendar",
                            key=f"cal_dl_{i}"
                        )
            
            # Copy response button — Base64 encoding prevents ANY quote/character clashing in HTML attributes
            b64_content = base64.b64encode(message["content"].encode("utf-8")).decode("utf-8")
            st.markdown(
                f'''<button onclick="(function(){{
                    var t=document.createElement('textarea');
                    t.value=atob('{b64_content}');
                    document.body.appendChild(t);
                    t.select();
                    document.execCommand('copy');
                    document.body.removeChild(t);
                    this.textContent='✅ Copied!';
                    setTimeout(()=>this.textContent='📋 Copy',2000);
                }}).call(this)"
                    style="background:transparent;border:1px solid rgba(255,255,255,0.15);
                           color:#aaa;border-radius:6px;padding:3px 10px;font-size:0.75rem;
                           cursor:pointer;margin-top:4px;">
                    📋 Copy
                </button>''',
                unsafe_allow_html=True
            )

        if message.get("tool_calls"):
            with st.expander("🛠️ Internal Logic & Tools"):
                for entry in message["tool_calls"]:
                    if "tool" in entry:
                        st.markdown(f"**Step:** Calling `{entry['tool']}`")
                        st.json(entry["args"])
                    elif "tool_result" in entry:
                        st.markdown(f"**Result from** `{entry['tool_result']}`:")
                        st.code(entry["output"], language=None)

# ---- Handle User Input (with voice + file support) ----
if prompt := st.chat_input("Ask me anything...", accept_file="multiple", accept_audio=True):
    
    # Process dynamically uploaded files
    text_input = prompt.text if hasattr(prompt, "text") and prompt.text else ""
    
    # Handle voice audio transcription
    if hasattr(prompt, "audio") and prompt.audio and not text_input:
        try:
            from groq import Groq
            import time
            audio_bytes = prompt.audio.read()
            tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_audio.write(audio_bytes)
            tmp_audio.close()
            
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            
            # Simple retry loop for voice transcription (Groq free tier has tight limits)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with open(tmp_audio.name, "rb") as af:
                        transcription = groq_client.audio.transcriptions.create(
                            model="whisper-large-v3",
                            file=af,
                            response_format="text"
                        )
                    text_input = transcription if isinstance(transcription, str) else transcription.text
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    raise e
            
            os.unlink(tmp_audio.name)
        except Exception as e:
            text_input = f"[Voice transcription failed: {str(e)}]"
    
    if not text_input and not (hasattr(prompt, "files") and prompt.files):
        text_input = "Please analyze my uploaded files."
    elif not text_input:
        text_input = "Please analyze my uploaded files."
    
    uploaded_files = prompt.files if hasattr(prompt, "files") and prompt.files else []
    
    internal_sys_prompt = ""
    docs_to_rag = []
    current_file_paths = []
    
    for f in uploaded_files:
        ext = os.path.splitext(f.name)[1].lower()
        
        # Save EVERY file to a temp location so it can be attached to emails
        # Use a safe filename from the upload
        safe_name = "".join(c for c in f.name if c.isalnum() or c in "._-").strip()
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex[:8]}_{safe_name}")
        with open(tmp_path, "wb") as tmp_f:
            tmp_f.write(f.getbuffer())
        current_file_paths.append(tmp_path)
        
        # Determine if it's an image or document
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            st.session_state.uploaded_image_path = tmp_path
            safe_path = json.dumps(tmp_path)
            internal_sys_prompt += (
                "\n\n[IMAGE ATTACHMENT]\n"
                f"- Path: {safe_path}\n"
                "- Tool: extract_text_from_image\n"
                "- Instruction: Run OCR on this image before answering image-content questions."
            )
            if ext in ['.webp']:
                internal_sys_prompt += "\n- Warning: WEBP OCR quality may vary by image compression."
            
        else:
            if "resume" in f.name.lower():
                # If the file is named resume, extract its text into session state for specific job tools
                try:
                    f.seek(0)  # Reset buffer position
                    text = ""
                    if ext == ".pdf":
                        pdf_reader = pypdf.PdfReader(io.BytesIO(f.read()))
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                    else:
                        text = f.read().decode("utf-8")
                    st.session_state.resume_text = sanitize_text(text)
                    internal_sys_prompt += f"\n\n[SYSTEM NOTIFICATION]: User uploaded a new resume. path: {tmp_path}"
                except Exception as e:
                    st.error(f"Failed to read resume for string extraction: {e}")
            
            # Normal document for RAG (All PDFs and TXTs go here, including resumes)
            if ext in ['.pdf', '.txt']:
                f.seek(0) # Reset buffer so PyPDFLoader can read it cleanly later
                docs_to_rag.append(f)

    if current_file_paths:
        internal_sys_prompt += "\n\n[AVAILABLE FILES FOR ATTACHMENT]:\n"
        for p in current_file_paths:
            internal_sys_prompt += f"- {os.path.basename(p)}: {p}\n"
        internal_sys_prompt += "If the user asks to 'send this file' or 'mail the resume', use these absolute paths in the `attachments` argument of `send_email`."
            
    if docs_to_rag:
        with st.spinner("Processing documents into Knowledge Base..."):
            try:
                st.session_state.vector_store = build_vector_store(docs_to_rag)
                internal_sys_prompt += f"\n\n[SYSTEM NOTIFICATION]: {len(docs_to_rag)} new document(s) added to the Knowledge Base. You can search them with the RAG tool."
            except Exception as e:
                st.error(f"Error processing documents: {e}")

    # Show user message
    messages.append({"role": "user", "content": text_input})
    
    # Update title if it's the first message
    if len(messages) == 1 and st.session_state.chat_sessions[current_id]["title"] == "New Chat":
        st.session_state.chat_sessions[current_id]["title"] = text_input[:30] + ("..." if len(text_input) > 30 else "")
        
    st.session_state.chat_sessions[current_id]["updated_at"] = datetime.now().isoformat()
    
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(text_input)
        if uploaded_files:
            st.caption(f"📎 Attached {len(uploaded_files)} file(s)")

    # Inject email credentials into the email tool before running agent
    set_email_context(
        st.session_state.get("user_email", ""),
        st.session_state.get("user_email_password", "")
    )
    
    # Generate streamed response
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            with st.spinner("Processing..."):
                # Use a dictionary to store mutable state from inside the generator
                stream_state = {"tool_calls_info": None, "final_response": ""}

                def process_stream():
                    resume_text = st.session_state.get("resume_text", "")
                    system_context = internal_sys_prompt
                        
                    vector_store = st.session_state.get("vector_store", None)
                    session_id = current_id
                    convo_summary = st.session_state.chat_sessions[current_id].get("summary", "")
                    
                    for event in action_agent_stream(
                        text_input, 
                        messages[:-1], 
                        resume_text=resume_text if resume_text else None,
                        extra_system_context=system_context if system_context else None,
                        vector_store=vector_store,
                        thread_id=session_id,
                        conversation_summary=convo_summary if convo_summary else None,
                    ):
                        if isinstance(event, dict) and event.get("type") == "tool_calls":
                            stream_state["tool_calls_info"] = event["data"]
                        elif isinstance(event, str):
                            stream_state["final_response"] += event
                            yield event
                
                # Write stream expects a generator of strings
                response = st.write_stream(process_stream())
                response = normalize_response_format(response)

            # Show tool calls in an expander
            tool_calls_info = stream_state["tool_calls_info"]
            if tool_calls_info:
                with st.expander("🛠️ Internal Logic & Tools", expanded=True):
                    for entry in tool_calls_info:
                        if "tool" in entry:
                            st.markdown(f"**Step:** Calling `{entry['tool']}`")
                            st.json(entry["args"])
                        elif "tool_result" in entry:
                            st.markdown(f"**Result from** `{entry['tool_result']}`:")
                            st.code(entry["output"], language=None)

            messages.append({
                "role": "assistant",
                "content": response,
                "tool_calls": tool_calls_info,
            })
            
            # Check if calendar event was created — store file path for download
            if response and "ICS_FILE_PATH:" in response:
                ics_match = re.search(r'ICS_FILE_PATH:(.+\.ics)', response)
                if ics_match:
                    st.session_state.pending_calendar_file = ics_match.group(1).strip()
            
            # Auto memory-summarize if conversation is too long
            if should_summarize(messages):
                with st.spinner("🧠 Summarizing memory..."):
                    st.session_state.chat_sessions[current_id] = compress_session(
                        st.session_state.chat_sessions[current_id]
                    )

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg or "quota" in error_msg.lower():
                st.error("📉 API Rate Limit Reached. The AI is a bit busy right now. Please wait 10-15 seconds and try again!")
            elif "api_key" in error_msg.lower() or "401" in error_msg or "api key not valid" in error_msg.lower():
                st.error("❌ Invalid API key. Please check your GEMINI_API_KEY in the .env file.")
            else:
                st.error(f"Something went wrong: {error_msg}")
