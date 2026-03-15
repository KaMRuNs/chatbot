"""
Action Agent — A ReAct-style agent that can use tools to perform real-world actions.

Uses LangGraph's create_react_agent to bind the LLM with custom tools
(email, etc.) and autonomously decide when to call them.
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from utils.llm import get_llm
from tools.email_tool import send_email
from tools.job_search_tool import search_jobs
from tools.skill_match_tool import match_skills
from tools.cover_letter_tool import generate_cover_letter
from tools.resume_optimizer_tool import optimize_resume
from tools.apply_to_job_tool import apply_to_job
from tools.calendar_tool import create_calendar_event
from tools.linkedin_tool import post_to_linkedin
from tools.unified_tools import internet_search, google_search_grounding, create_retrieval_tool
from tools.ocr_tool import extract_text_from_image

SYSTEM_PROMPT = (
    "You are a sophisticated personal assistant designed to help users with information retrieval and tasks.\n\n"
    "CORE CAPABILITIES:\n"
    "1. **Internet Search**: Use `internet_search` for general queries via DuckDuckGo.\n"
    "2. **Google Search**: Use `google_search_grounding` for highly factual or grounded information.\n"
    "3. **Document Search**: Use `search_uploaded_documents` to query the user's uploaded files (PDF/TXT).\n"
    "4. **Image OCR**: Use `extract_text_from_image` to read text from uploaded images.\n"
    "5. **Professional Tools**: Tools to match skills (`match_skills`), optimize resumes (`optimize_resume`), generate cover letters (`generate_cover_letter`), and search for jobs (`search_jobs`).\n"
    "6. **Job Applications**: Use `apply_to_job` to automatically search and apply for roles.\n"
    "7. **Actions**: Send emails (`send_email`), schedule events (`create_calendar_event`), or post to LinkedIn (`post_to_linkedin`).\n\n"
    "STRICT TOOL CALLING RULES:\n"
    "- ALWAYS use the built-in tool calling mechanism provided by the API.\n"
    "- DO NOT output tool names or arguments as plain text or XML (e.g., `<function=name>{...}</function>`).\n"
    "- Output tool calls IMMEDIATELY without any conversational filler or prefaces.\n\n"
    "OPERATIONAL RULES:\n"
    "- If a question can be answered from docs, use `search_uploaded_documents`.\n"
    "- If a question is about current events, use `internet_search`.\n"
    "- If the user says 'hi' or just chats, respond naturally without using tools.\n"
    "- IMPORTANT: Provide all required arguments for tools from the context provided below.\n"
    "- Be helpful, concise, and professional."
)

# Base tools
BASE_TOOLS = [
    send_email,
    search_jobs, match_skills, generate_cover_letter, optimize_resume,
    apply_to_job, create_calendar_event, post_to_linkedin,
    internet_search, google_search_grounding,
    extract_text_from_image
]


def action_agent_stream(
    query: str, 
    chat_history: list = None, 
    resume_text: str = None, 
    vector_store = None,
    thread_id: str = "default_thread",
    conversation_summary: str = None,
):
    """Runs the unified smart agent and streams output."""
    llm = get_llm()

    messages = []
    if chat_history:
        for msg in chat_history[-6:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=query))

    current_system_prompt = SYSTEM_PROMPT
    if conversation_summary:
        current_system_prompt += f"\n\nEarlier Summary: {conversation_summary}"
    if resume_text:
        current_system_prompt += f"\n\nUser Data Context:\n{resume_text}"

    rag_tool = create_retrieval_tool(vector_store)
    current_tools = list(BASE_TOOLS) + [rag_tool]

    agent = create_react_agent(
        model=llm,
        tools=current_tools,
        prompt=current_system_prompt,
    )

    tool_calls_log = []
    final_response = ""

    # Use 'updates' mode for stable result extraction
    for event in agent.stream({"messages": messages}, stream_mode="updates"):
        for node_name, node_output in event.items():
            msgs = node_output.get("messages", [])
            if not isinstance(msgs, list): msgs = [msgs]
            
            for msg in msgs:
                # 1. Capture tool calls from the agent
                if node_name == "agent" and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_log.append({"tool": tc["name"], "args": tc["args"]})
                
                # 2. Capture results from tools
                if node_name == "tools" and msg.type == "tool":
                    tool_calls_log.append({"tool_result": msg.name, "output": msg.content})

                # 3. Capture final text from the agent
                # AI messages without tool calls are the final answer
                if node_name == "agent" and msg.type == "ai" and msg.content:
                    if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                        final_response = msg.content

        # Yield tool log to update UI expander
        if tool_calls_log:
            yield {"type": "tool_calls", "data": list(tool_calls_log)}

    # Finally yield character-by-character for st.write_stream
    if final_response:
        for char in final_response:
            yield char
