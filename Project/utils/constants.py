"""
Application-wide constants shared across modules.
"""

SYSTEM_NAME = "Agentic RAG Knowledge Assistant"

TOOL_NAMES = {
    "document_search": "document_search",
    "summarization": "summarization",
    "calculator": "calculator",
    "csv_analysis": "csv_analysis",
    "excel_analysis": "excel_analysis",
}

FILE_TYPE_PDF = "pdf"
FILE_TYPE_DOCX = "docx"
FILE_TYPE_TXT = "txt"
FILE_TYPE_CSV = "csv"
FILE_TYPE_XLSX = "xlsx"

DEFAULT_ANSWER_NO_CONTEXT = (
    "I could not find relevant information in the knowledge base to answer this question."
)

MEMORY_WINDOW_SIZE = 10
