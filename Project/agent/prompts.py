"""
Prompt templates used by the agent for planning, query rewriting,
and final answer generation.
"""

PLANNER_PROMPT = """You are a planning module inside an agentic RAG system.
Given the conversation history and the user's latest question, decide:
1. Whether document retrieval is required to answer the question.
2. Whether any additional tools are required (calculator, csv_analysis,
   excel_analysis, summarization).
3. A short, clear search query to use if retrieval is required (rewrite
   the user's question into a focused, standalone query using context
   from the conversation if needed).

Conversation history:
{history}

User question:
{question}

Respond ONLY with a JSON object in this exact format, no extra text:
{{
  "needs_retrieval": true or false,
  "tools": ["tool_name", ...],
  "search_query": "the rewritten search query"
}}
"""

ANSWER_PROMPT = """You are a precise, helpful knowledge assistant. Answer the
user's question using ONLY the context provided below and the conversation
history. If the context does not contain the answer, say so honestly.

Always cite the source document names you used in your answer, in the
format (Source: filename).

If multiple documents contain relevant information, compare and synthesize
across them clearly.

Conversation history:
{history}

Retrieved context:
{context}

Tool outputs (if any):
{tool_outputs}

User question:
{question}

Answer:
"""

SUMMARIZATION_PROMPT = """Summarize the following text concisely while
preserving all key facts, figures, and named entities. Keep the summary
proportional to the length and complexity of the input.

Text:
{text}

Summary:
"""

QUERY_REWRITE_PROMPT = """Rewrite the following user question into a
clear, standalone search query suitable for semantic document retrieval.
Use the conversation history to resolve pronouns or references.

Conversation history:
{history}

User question:
{question}

Rewritten query:
"""
