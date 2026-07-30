# Agentic RAG Knowledge Assistant

A production-style **Agentic Retrieval-Augmented Generation** system that
plans its reasoning steps, decides when retrieval and tools are needed,
searches a local FAISS knowledge base, and answers complex questions with
cited sources — all through a Streamlit chat interface.

## Overview

- **Orchestration:** LangChain + LangGraph (planning, retrieval, tool use, answer generation)
- **LLM:** Google Gemini (via `langchain-google-genai`)
- **Vector store:** FAISS (persisted locally)
- **Embeddings:** Sentence Transformers `all-MiniLM-L6-v2`
- **Supported documents:** PDF, DOCX, TXT, CSV, XLSX
- **Interface:** Streamlit

### How the agent works

1. **Plan** – The planner node decides if retrieval is required, which
   tools (if any) are needed, and rewrites the user's question into a
   focused search query.
2. **Retrieve** – If needed, the retriever fetches the most relevant
   chunks from the FAISS index.
3. **Tools** – The agent can call the Calculator, Summarization, CSV
   Analysis, or Excel Analysis tools depending on the request.
4. **Answer** – The LLM generates a final answer using retrieved context,
   tool outputs, and conversation history, always citing sources.

## Project Structure

```
project/
├── app.py                 # Streamlit UI entry point
├── config.py               # Centralized configuration
├── requirements.txt
├── README.md
├── .env.example             # Template for environment variables
├── documents/                # Put your source documents here
├── vector_store/              # Persisted FAISS index (auto-generated)
├── data/                      # Logs and misc runtime data
├── agent/
│   ├── agent.py               # LangGraph agent workflow
│   ├── prompts.py             # Prompt templates
│   ├── memory.py               # Conversation memory
│   └── tools.py                # Agent tools
├── rag/
│   ├── embeddings.py            # Embedding model wrapper
│   ├── retriever.py              # Retrieval interface
│   ├── chunking.py                # Semantic chunking
│   ├── loaders.py                  # Document loaders (PDF/DOCX/TXT/CSV/XLSX)
│   └── vector_database.py           # FAISS build/load/persist
├── utils/
│   ├── helpers.py
│   ├── constants.py
│   └── logger.py
└── assets/
```

## Installation

### 1. Prerequisites

- Python 3.11
- Visual Studio Code (recommended)
- A Google Gemini API key: https://aistudio.google.com/app/apikey

### 2. Clone / open the project in VS Code

Open the `project/` folder in Visual Studio Code.

### 3. Create and activate a virtual environment

**Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

Edit `.env`:
```
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### 6. Add your documents

Place your PDF, DOCX, TXT, CSV, or XLSX files inside the `documents/`
folder (or upload them later from the Streamlit sidebar).

## Running the Application

From the project root, with the virtual environment activated:

```bash
streamlit run app.py
```

Streamlit will open the app in your browser, typically at:
```
http://localhost:8501
```

## Usage Guide

1. **Add documents** – Use the sidebar file uploader, or drop files
   directly into the `documents/` folder.
2. **Build the knowledge base** – Click **"🔄 Rebuild Knowledge Base"**
   in the sidebar. This extracts, cleans, chunks, embeds, and saves the
   FAISS index to `vector_store/`.
3. **Ask questions** – Type a question in the chat box at the bottom.
   The agent will plan, retrieve relevant chunks, optionally use tools,
   and answer with cited sources.
4. **Follow-up questions** – The agent remembers recent conversation
   turns, so you can ask follow-ups without repeating context.
5. **Clear conversation** – Use the sidebar button to reset the chat
   and memory.
6. **Restart the app** – The FAISS index is persisted on disk and is
   automatically reloaded the next time you start the app, so you don't
   need to rebuild it every session unless documents changed.

### Example questions

- "Summarize the key findings across all uploaded reports."
- "Compare the revenue figures in report_a.pdf and report_b.docx."
- "What is 245 * 12 / 4?" (uses the calculator tool)
- "Analyze sales_data.csv and tell me the average order value."

## Notes on Tools

| Tool               | Purpose                                              |
|--------------------|-------------------------------------------------------|
| `document_search`  | Semantic search over the FAISS knowledge base          |
| `summarization`    | Summarizes long text passages                            |
| `calculator`        | Evaluates arithmetic expressions safely                    |
| `csv_analysis`      | Summary statistics for a CSV file in `documents/`           |
| `excel_analysis`    | Summary statistics for each sheet of an XLSX file             |

## Troubleshooting

- **`GOOGLE_API_KEY` errors:** Make sure `.env` exists and contains a
  valid key, and that you restarted Streamlit after editing it.
- **Empty knowledge base:** Click "Rebuild Knowledge Base" after adding
  documents — the index is not built automatically on file upload.
- **Slow first run:** The first run downloads the `all-MiniLM-L6-v2`
  embedding model; subsequent runs use the local cache.

## License

This project is provided as-is for internal / personal use. Adapt as needed.
