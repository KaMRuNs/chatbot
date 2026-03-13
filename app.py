import streamlit as st
import json
import os
from dotenv import load_dotenv

# Load environment variables at the very beginning
load_dotenv(override=True)

from agents.rag_agent import build_vector_store
from agents.action_agent import action_agent_stream
import pypdf
import io

# ---- Page Config ----
st.set_page_config(page_title="Simple Chatbot", page_icon="🤖", layout="centered")

# ---- Custom CSS ----
st.markdown("""
<style>
    /* Import modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background & Glassmorphism elements */
    .stApp {
        background: radial-gradient(circle at 15% 50%, #1e1b4b, #0f172a 60%);
    }

    /* Top Title Styling */
    h1 {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Agent Badges with glowing & glass logic */
    .agent-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .badge-search {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(56, 189, 248, 0.05));
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38bdf8;
    }
    
    .badge-rag {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.2), rgba(52, 211, 153, 0.05));
        border: 1px solid rgba(52, 211, 153, 0.4);
        color: #34d399;
    }
    
    .badge-action {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(251, 191, 36, 0.05));
        border: 1px solid rgba(251, 191, 36, 0.4);
        color: #fbbf24;
    }
    
    /* Sleek Chat Messages */
    .stChatMessage {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 12px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }

    /* Sidebar info styling */
    .sidebar-info {
        font-size: 0.85rem;
        color: #94a3b8;
        padding: 12px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 16px;
    }

    /* RAG Chunks styling */
    .chunk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-weight: 500;
        color: #e2e8f0;
    }
    
    .chunk-meta {
        font-size: 0.75rem;
        color: #cbd5e1;
        background: rgba(15, 23, 42, 0.6);
        padding: 3px 8px;
        border-radius: 6px;
    }

    /* Inputs & Buttons glow effects */
    .stTextInput div[data-baseweb="input"], .stChatInputContainer {
        border-radius: 12px !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput div[data-baseweb="input"]:focus-within, .stChatInputContainer:focus-within {
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.5) !important;
    }
    
    div[data-testid="stButton"] button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    div[data-testid="stButton"] button:hover {
        transform: scale(1.02);
    }

</style>
""", unsafe_allow_html=True)

st.title("Simple Chatbot")


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
    st.header("Settings")

    st.markdown('<p class="sidebar-info">Powered by GPT-OSS-120B via Groq</p>', unsafe_allow_html=True)

    st.divider()

    # Document Upload Section
    st.subheader("📁 Knowledge Base")
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload files", type=["pdf", "txt"], accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    try:
                        st.session_state.vector_store = build_vector_store(uploaded_files)
                        st.success(f"Processed {len(uploaded_files)} file(s)!")
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")
                        
    # Resume Upload Section
    st.subheader("📄 Resume (Optional)")
    st.markdown("Upload your resume to use job matching tools.")
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "txt"], key="resume_uploader")
        
        if resume_file:
            if st.button("Extract Resume Text"):
                with st.spinner("Extracting..."):
                    try:
                        text = ""
                        if resume_file.name.endswith(".pdf"):
                            pdf_reader = pypdf.PdfReader(io.BytesIO(resume_file.read()))
                            for page in pdf_reader.pages:
                                text += page.extract_text() + "\n"
                        else:
                            text = resume_file.read().decode("utf-8")
                            
                        st.session_state.resume_text = text
                        st.success("Resume extracted successfully!")
                    except Exception as e:
                        st.error(f"Error extracting resume: {e}")
        
        if "resume_text" in st.session_state and st.session_state.resume_text:
            st.success("Resume loaded and ready for action.")
            with st.expander("View Extracted Text"):
                st.text(st.session_state.resume_text[:500] + "...")

    # Status Info
    if "vector_store" in st.session_state:
        st.success("✅ Documents indexed and ready for Q&A.")
    else:
        st.warning("💡 Upload docs to enable the RAG tool.")
    
    st.info("🛠️ Assistant can also search the web, send emails, and set alarms automatically.")

    st.divider()

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Export chat button
    if "messages" in st.session_state and st.session_state.messages:
        chat_export = ""
        for msg in st.session_state.messages:
            role = "You" if msg["role"] == "user" else "Assistant"
            chat_export += f"**{role}:**\n{msg['content']}\n\n---\n\n"
        st.download_button(
            "Export Chat",
            data=chat_export,
            file_name="smart_assistant_chat.md",
            mime="text/markdown",
        )

# ---- Initialize Chat History ----
if "messages" not in st.session_state:
    st.session_state.messages = []

messages = st.session_state.messages

# ---- Poll for Fired Alarms ----
ALARM_FILE = os.path.join(os.path.dirname(__file__), "pending_alarms.json")
if os.path.exists(ALARM_FILE):
    try:
        with open(ALARM_FILE, "r", encoding="utf-8") as f:
            alarms = json.load(f)
        if alarms:
            for alarm in alarms:
                st.toast(f"⏰ Alarm: {alarm['message']}", icon="⏰")
            # Clear the file after showing
            with open(ALARM_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
    except (json.JSONDecodeError, KeyError):
        pass

# ---- Display Chat History ----
for message in messages:
    with st.chat_message(message["role"]):
        # Show agent badge on assistant messages
        if message["role"] == "assistant":
            badge_class = "badge-action"
            st.markdown(
                f'<span class="agent-badge {badge_class}">Smart Assistant</span>',
                unsafe_allow_html=True,
            )
        st.markdown(message["content"])

        # Show tool info if available
        if message.get("tool_calls"):
            with st.expander("🛠️ Internal Logic & Tools"):
                for entry in message["tool_calls"]:
                    if "tool" in entry:
                        st.markdown(f"**Step:** Calling `{entry['tool']}`")
                        st.json(entry["args"])
                    elif "tool_result" in entry:
                        st.markdown(f"**Result from** `{entry['tool_result']}`:")
                        st.code(entry["output"], language=None)

# ---- Handle User Input ----
if prompt := st.chat_input("Ask me anything..."):
    # Show user message
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate streamed response
    with st.chat_message("assistant"):
        # Show agent badge
        st.markdown(
            f'<span class="agent-badge badge-action">Smart Assistant</span>',
            unsafe_allow_html=True,
        )

        try:
            with st.spinner("🤔 Assistant is thinking and acting..."):
                # Use a dictionary to store mutable state from inside the generator
                stream_state = {"tool_calls_info": None, "final_response": ""}

                def process_stream():
                    resume_text = st.session_state.get("resume_text", None)
                    vector_store = st.session_state.get("vector_store", None)
                    # Use Streamlit session ID for consistent LangGraph thread ID
                    session_id = st.runtime.scriptrunner.get_script_run_ctx().session_id
                    
                    for event in action_agent_stream(
                        prompt, 
                        messages[:-1], 
                        resume_text=resume_text, 
                        vector_store=vector_store,
                        thread_id=session_id
                    ):
                        if isinstance(event, dict) and event.get("type") == "tool_calls":
                            stream_state["tool_calls_info"] = event["data"]
                        elif isinstance(event, str):
                            stream_state["final_response"] += event
                            yield event
                
                # Write stream expects a generator of strings
                response = st.write_stream(process_stream())

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

            # We don't need st.markdown(response) here anymore since write_stream rendered it
            messages.append({
                "role": "assistant",
                "content": response,
                "tool_calls": tool_calls_info,
            })

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                st.error("Rate limit reached. Please wait a moment and try again.")
            elif "api_key" in error_msg.lower() or "401" in error_msg:
                st.error("Invalid API key. Please check your GROQ_API_KEY in the .env file.")
            else:
                st.error(f"Something went wrong: {error_msg}")
