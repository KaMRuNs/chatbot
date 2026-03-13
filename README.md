# Simple Chatbot - LangChain + Streamlit

Welcome! In this project, you will build a sophisticated **Unified Smart Assistant** that can search the web, answer questions from documents (RAG), and perform real-world actions like sending emails or setting alarms—all within a single, persistent conversation using **LangGraph**.

## What You Will Learn

Through this project, you will learn:

- How to connect to a Large Language Model (LLM) using LangChain
- How to orchestrate agent workflows using **LangGraph**
- How to give an LLM the ability to search the web and perform actions
- How to build a RAG (Retrieval-Augmented Generation) pipeline from scratch
- How to implement conversation persistence with checkpoints
- How to trace and debug agent workflows with **LangSmith**
- How to create a premium chat interface with Streamlit

Let me walk you through everything step by step.

## Understanding the Architecture

Before we start coding, let me explain how each agent works.

### Search Agent

When you ask a question, here is what happens behind the scenes:

```
Your Question
    ↓
LLM receives it with browser_search tool enabled
    ↓
LLM searches the web, reads relevant pages
    ↓
LLM writes a concise answer with source links
```

You don't need to configure any search API. The model (GPT-OSS-120B) has a built-in browser search tool provided by Groq. We just enable it.

### RAG Agent

RAG stands for **Retrieval-Augmented Generation**. Instead of letting the LLM answer from its training data, we feed it your specific documents. Here is the flow:

```
Step 1 — Document Processing (happens once when you upload):

Your PDF/TXT files
    ↓
Split into small chunks (~500 characters each)
    ↓
Each chunk is converted into a vector (embedding)
    ↓
All vectors are stored in FAISS (a local vector database)


Step 2 — Answering (happens every time you ask):

Your Question
    ↓
Convert question into a vector
    ↓
Find the 3 most similar chunks from FAISS
    ↓
Send those chunks + your question to the LLM
    ↓
LLM answers based on the document content
```

Now let me explain a few terms you need to know.

## Key Concepts

**What is an embedding?**
Think of it as converting text into a list of numbers that captures its meaning. The sentence "I am happy" and "I feel joyful" would produce similar numbers, because they mean similar things. We use the `all-MiniLM-L6-v2` model to generate these embeddings — it runs on your computer, no API key needed.

**What is FAISS?**
FAISS (Facebook AI Similarity Search) is a library that stores vectors and lets you quickly find which ones are most similar to a given query vector. We use it as our vector database.

**What is streaming?**
Normally, you send a question and wait for the complete answer. With streaming, you receive the answer token by token as the LLM generates it. This gives a much better user experience — the user sees text appearing in real time instead of staring at a loading spinner.

**What is `reasoning_effort="low"`?**
GPT-OSS-120B is a reasoning model — it "thinks" before answering by generating internal reasoning tokens. Setting this to `low` reduces those hidden tokens, which saves your free-tier quota. For most chatbot questions, low reasoning is more than enough.

## Project Structure

Take a look at how the project is organized:

```
simple_chatbot/
├── .env                      # Your API key goes here
├── .streamlit/
│   └── config.toml           # Theme configuration (dark mode)
├── requirements.txt          # All Python packages we need
├── app.py                    # Main file — Streamlit UI
├── agents/
│   ├── __init__.py
│   ├── search_agent.py       # Web search agent
│   └── rag_agent.py          # Document Q&A agent
└── utils/
    ├── __init__.py
    └── llm.py                # LLM connection setup
```

I have separated things into folders so each file has a single responsibility:

- **`utils/llm.py`** — This is where we configure the LLM. Both agents import from here, so if you ever want to change the model or settings, you only change one file.

- **`agents/search_agent.py`** — This file handles web search. It sends your question to the LLM with the `browser_search` tool enabled, cleans up the response, and streams it back.

- **`agents/rag_agent.py`** — This file handles document Q&A. It has three functions: one to process your uploaded files, one to answer questions, and one to show you which document chunks were retrieved.

- **`app.py`** — This is the main entry point. It builds the Streamlit UI, manages chat history, handles agent switching, and displays responses.

## Tech Stack

Here is what we are using and why:

| Component | Tool | Why This One |
|-----------|------|--------------|
| LLM | GPT-OSS-120B via Groq | Fast reasoning model, has built-in web search |
| Orchestration | LangGraph | Industry standard for building persistent agentic workflows |
| Embeddings | all-MiniLM-L6-v2 | Runs locally, free, no API key needed |
| Vector Store | FAISS | Runs locally, high-performance similarity search |
| Tracing | LangSmith | Essential for debugging agent internal states and tool calls |
| UI | Streamlit | Premium Python web UI with custom glassmorphism design |

## Submission Requirements Checklist

Before submitting, ensure you have:
1. **LangGraph implementation** (found in `agents/action_agent.py`)
2. **Two or more functional tools** (check the `tools/` folder)
3. **LangSmith Tracing** enabled in `.env`
4. **Conversation persistence** implemented via `MemorySaver`
5. **Demo Video** (3-5 minutes)

You only need **one API key** (Groq). Everything else runs on your machine.

## Setup Instructions

Follow these steps carefully.

### Step 1 — Get Your API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Create an account or log in
3. Navigate to **API Keys** in the left sidebar
4. Click **Create API Key** and copy it

Keep this key safe. Do not share it publicly.

### Step 2 — Install Dependencies

Open your terminal, navigate to the project folder, and run:

```bash
cd simple_chatbot

# I recommend creating a virtual environment first
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Now install all packages
pip install -r requirements.txt
```

This will install LangChain, Streamlit, FAISS, and all other required packages. The embedding model (~80MB) will download automatically on first run.

### Step 3 — Add Your API Key

Open the `.env` file and paste your key:

```
GROQ_API_KEY=your_actual_key_here
```

Replace `your_actual_key_here` with the key you copied from Groq.

### Step 4 — Run the App

```bash
streamlit run app.py
```

Your browser will open at `http://localhost:8501`. You should see the chatbot interface.

## How to Use the Smart Assistant

1. **Upload Documents**: In the sidebar, upload your PDFs or text files under "Knowledge Base" and click **Process Documents**.
2. **Upload Resume**: Optionally upload your resume to enable job matching and resume optimization.
3. **Chat Naturally**: Just ask the assistant anything! You don't need to switch modes. 
   - Try: *"What happened in the news today?"* (Triggers Internet Search)
   - Try: *"Summarize my uploaded document."* (Triggers Document Search)
   - Try: *"Send an email to boss@example.com about the report."* (Triggers Email Tool)
4. **View Logic**: Expand **Internal Logic & Tools** in the chat to see exactly how the agent decided which tool to use.

## Troubleshooting

If something goes wrong, check here first:

| Problem | What to Do |
|---------|------------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `Invalid API key` error | Open `.env` and make sure your `GROQ_API_KEY` is correct |
| `Rate limit reached` | You are on the free tier. Wait 1-2 minutes and try again |
| App does not start | Make sure you are inside the `simple_chatbot` folder |
| PDF not loading | Check that the file is not corrupted or password-protected |
| Embedding model download is slow | This only happens the first time. Be patient, it is ~80MB |

## What to Explore Next

Once you are comfortable with this project, here are some challenges for you:

1. **Add more file types** — Try supporting DOCX or CSV files in the RAG agent
2. **Try a different embedding model** — Swap `all-MiniLM-L6-v2` with a larger model and compare results
3. **Deploy to Streamlit Cloud** — Make your chatbot accessible to anyone on the internet
4. **Add a third agent** — For example, a code generation agent or a summarization agent
5. **Explore LangGraph** — Rebuild this project using LangGraph for multi-step agent workflows

Good luck, and feel free to experiment!
