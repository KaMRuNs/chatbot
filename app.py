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
    /* Gemini-inspired Dark Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #131314;
    }

    .stApp {
        background-color: #131314;
    }

    /* Minimalist SideBar */
    section[data-testid="stSidebar"] {
        background-color: #1e1f20 !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Modern Typography */
    h1 {
        font-family: 'Outfit', sans-serif;
        color: #e3e3e3;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0px !important;
    }

    .gemini-subheader {
        font-size: 1.2rem;
        color: #8e918f;
        margin-bottom: 2rem;
    }

    /* Agent Badges (Gemini Style) */
    .agent-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 12px;
        background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570);
        color: white;
        box-shadow: 0 0 15px rgba(66, 133, 244, 0.3);
    }

    /* Chat Message Bubbles */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 20px 0 !important;
        max-width: 800px;
        margin: 0 auto;
    }

    /* User Message Style */
    div[data-testid="stChatMessageUser"] {
        background-color: #1e1f20 !important;
        border-radius: 24px !important;
        padding: 12px 20px !important;
        margin-left: auto;
        width: fit-content;
        max-width: 80%;
    }

    /* Assistant Message Style */
    div[data-testid="stChatMessageAssistant"] {
        padding-left: 0 !important;
    }

    /* Chat Input Fixed to Bottom */
    .stChatInputContainer {
        border-radius: 30px !important;
        background-color: #1e1f20 !important;
        border: 1px solid #444746 !important;
        padding: 5px 15px !important;
        transition: border-color 0.3s ease;
    }

    .stChatInputContainer:focus-within {
        border-color: #8ab4f8 !important;
    }

    /* Hide redundant elements */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #131314;
    }
    ::-webkit-scrollbar-thumb {
        background: #444746;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }

    /* Buttons */
    div[data-testid="stButton"] button {
        background-color: #1e1f20;
        border: 1px solid #444746;
        color: #e3e3e3;
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
    }

    div[data-testid="stButton"] button:hover {
        background-color: #333537;
        border-color: #8ab4f8;
    }

</style>
""", unsafe_allow_html=True)

st.title("Hello, User")
st.markdown('<div class="gemini-subheader">How can I help you today?</div>', unsafe_allow_html=True)


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
    st.header("Assistant Control")
    st.divider()

    # Document Upload Section
    st.subheader("📁 Knowledge Base")
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

    # Status & Legend
    st.markdown("---")
    if "vector_store" in st.session_state:
        st.caption("✅ Knowledge Base Active")
    else:
        st.caption("💡 Knowledge Base Empty")
    
    if "resume_text" in st.session_state:
        st.caption("✅ Resume Background Active")

    st.caption("🛠️ Web, Email, & Alarms are always ON")

    st.divider()

    # Clear chat button
    if st.button("Clear Chat", use_container_width=True):
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
            use_container_width=True
        )

    # Move technical info to bottom
    st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-info">Engine: GPT-OSS-120B<br>Model: Groq Llama3-70b-Tool-Use</p>', unsafe_allow_html=True)

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
            st.markdown(
                f'<span class="agent-badge">Smart Assistant</span>',
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
        st.markdown(
            f'<span class="agent-badge">Smart Assistant</span>',
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
