"""
Action Agent — A ReAct-style agent that can use tools to perform real-world actions.

Uses LangGraph's create_react_agent to bind the LLM with custom tools
(email, alarm, etc.) and autonomously decide when to call them.
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from utils.llm import get_llm
from tools.email_tool import send_email
from tools.alarm_tool import set_alarm
from tools.job_search_tool import search_jobs
from tools.skill_match_tool import match_skills
from tools.cover_letter_tool import generate_cover_letter
from tools.resume_optimizer_tool import optimize_resume
from tools.unified_tools import internet_search, create_retrieval_tool

SYSTEM_PROMPT = (
    "You are a sophisticated personal assistant that can search the web, answer questions from documents, and perform real-world actions.\n\n"
    "You have access to the following capabilities:\n"
    "1. **Internet Search**: Use `internet_search` for current events, news, or general information not in the docs.\n"
    "2. **Document Search**: Use `search_documents` to find information specifically within the user's uploaded PDF or text files.\n"
    "3. **Real-world Actions**: Send emails, set alarms, search jobs, analyze skills, and optimize resumes.\n\n"
    "Guidelines:\n"
    "- If a user asks about their files, ALWAYS try `search_documents` first.\n"
    "- If they ask about current events, use `internet_search`.\n"
    "- If they ask to perform an action (email, alarm), use the specific action tool.\n"
    "- If you need missing info to run a tool, ask the user.\n"
    "- Be concise and friendly. Confirm your actions clearly."
)

# Base tools
BASE_TOOLS = [send_email, set_alarm, search_jobs, match_skills, generate_cover_letter, set_alarm, optimize_resume, internet_search]


# Checkpointer for conversation persistence
memory = MemorySaver()


def action_agent_stream(
    query: str, 
    chat_history: list = None, 
    resume_text: str = None, 
    vector_store = None,
    thread_id: str = "default_thread"
):
    """Runs the unified smart agent and streams the final response token by token."""
    llm = get_llm()

    # Build message history
    messages = []

    if chat_history:
        for msg in chat_history[-6:]:  # Last 3 exchanges
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=query))
    
    # Inject resume context into system prompt if available
    current_system_prompt = SYSTEM_PROMPT
    if resume_text:
        current_system_prompt += (
            f"\n\nThe user has uploaded their resume. ALWAYS use this resume text "
            f"when asked to match skills, write a cover letter, or optimize their resume:\n\n"
            f"<RESUME_TEXT>\n{resume_text}\n</RESUME_TEXT>\n"
            "Pass this exact resume text to the resume/job related tools when needed."
        )

    # Dynamically add RAG tool if vector store is available
    current_tools = list(BASE_TOOLS)
    if vector_store:
        rag_tool = create_retrieval_tool(vector_store)
        current_tools.append(rag_tool)

    # Create the ReAct agent using LangGraph
    agent = create_react_agent(
        model=llm,
        tools=current_tools,
        prompt=current_system_prompt,
        checkpointer=memory,
    )

    config = {"configurable": {"thread_id": thread_id}}

    # Run the agent and collect tool call info + final response
    tool_calls_log = []
    final_response = ""

    for event in agent.stream({"messages": messages}, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            messages_output = node_output["messages"]
            if not isinstance(messages_output, list):
                messages_output = [messages_output]

            for msg in messages_output:
                # Capture tool calls made by the agent
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_log.append({
                            "tool": tc["name"],
                            "args": tc["args"],
                        })

                # Capture tool results
                if msg.type == "tool":
                    tool_calls_log.append({
                        "tool_result": msg.name,
                        "output": msg.content,
                    })

                # Stream the final AI response
                if msg.type == "ai" and msg.content and not msg.tool_calls:
                    final_response = msg.content

    # Yield tool call info first (for the UI to show in an expander)
    if tool_calls_log:
        yield {"type": "tool_calls", "data": tool_calls_log}

    # Then yield the final response token by token for st.write_stream
    if final_response:
        # Yield character by character for streaming effect
        for char in final_response:
            yield char
