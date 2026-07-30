# 🤖 Agentic RAG System

An intelligent **Agentic Retrieval-Augmented Generation (Agentic RAG)** system that answers questions from a custom knowledge base using AI reasoning, semantic search, and document retrieval. The application supports multiple document formats, retrieves relevant information from uploaded files, and generates accurate answers with source citations through a modern **Streamlit** interface.

---

## 🚀 Features

- Upload and manage your own knowledge base.
- Supports **PDF, DOCX, TXT, CSV, and XLSX** files.
- Automatic document processing and text extraction.
- Text cleaning and intelligent chunking.
- Semantic search using vector embeddings.
- FAISS vector database for fast retrieval.
- AI-powered Agent capable of reasoning before answering.
- Multi-step question answering.
- Automatic query refinement.
- Document comparison and summarization.
- CSV and Excel data analysis.
- Built-in calculator tool.
- Conversation memory for follow-up questions.
- Source citation for every generated answer.
- Modern Streamlit web interface.

---

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- LangGraph
- Google Gemini
- FAISS
- Sentence Transformers
- Pandas
- PyMuPDF
- python-docx
- OpenPyXL

---

## 📂 Supported File Types

- PDF
- DOCX
- TXT
- CSV
- XLSX

---

## 🧠 How It Works

1. Upload documents to the **documents** folder or through the web interface.
2. Extract text from supported document formats.
3. Clean and split the text into chunks.
4. Generate embeddings using Sentence Transformers.
5. Store embeddings inside a FAISS vector database.
6. The AI Agent analyzes the user's question.
7. The Agent decides which tools to use.
8. Relevant document chunks are retrieved.
9. Gemini generates an answer based on the retrieved context.
10. The final response includes the source documents.

---

## 📁 Project Structure

```text
project/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .env.example
│
├── documents/
├── vector_store/
├── data/
├── assets/
│
├── agent/
│   ├── agent.py
│   ├── memory.py
│   ├── prompts.py
│   └── tools.py
│
├── rag/
│   ├── loaders.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retriever.py
│   └── vector_database.py
│
└── utils/
    ├── helpers.py
    ├── constants.py
    └── logger.py
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/agentic-rag-system.git
cd agentic-rag-system
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---



## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to:

```text
http://localhost:8501
```

---

## 💬 Example Questions

- What is Machine Learning?
- Summarize Chapter 3.
- Compare Deep Learning and Machine Learning.
- Analyze the uploaded CSV file.
- What is the average value in the Excel sheet?
- Explain the AI lifecycle.
- Which document discusses neural networks?

---

## 📚 Learning Outcomes

- Retrieval-Augmented Generation (RAG)
- Agentic AI
- Semantic Search
- Vector Databases
- Embeddings
- LangChain & LangGraph
- Document Processing
- Streamlit Application Development


## 📄 License

This project was developed for educational and learning purposes.
````
