"""
Tool definitions for the agentic RAG system.
Each tool is exposed as a LangChain @tool function so the agent can
select and invoke them dynamically based on the user's request.
"""

import ast
import operator
import os
from typing import List

import pandas as pd
from langchain_core.tools import tool

from config import settings
from rag.retriever import Retriever
from agent.prompts import SUMMARIZATION_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    """Recursively evaluate a restricted arithmetic AST node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported or unsafe expression.")


class ToolFactory:
    """Builds LangChain tool instances bound to a shared retriever and LLM."""

    def __init__(self, retriever: Retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def build_all(self) -> List:
        """Return the full list of tool instances used by the agent."""
        return [
            self.document_search_tool(),
            self.summarization_tool(),
            self.calculator_tool(),
            self.csv_analysis_tool(),
            self.excel_analysis_tool(),
        ]

    def document_search_tool(self):
        retriever = self.retriever

        @tool("document_search")
        def document_search(query: str) -> str:
            """Search the document knowledge base for chunks relevant to a query.
            Use this whenever the user asks about content from uploaded documents."""
            results = retriever.retrieve(query)
            if not results:
                return "No relevant documents found."
            formatted = []
            for item in results:
                formatted.append(
                    f"Source: {item['source']} | Content: {item['content']}"
                )
            return "\n---\n".join(formatted)

        return document_search

    def summarization_tool(self):
        llm = self.llm

        @tool("summarization")
        def summarization(text: str) -> str:
            """Summarize a long piece of text into a concise summary while
            preserving key facts, figures, and names."""
            prompt = SUMMARIZATION_PROMPT.format(text=text)
            response = llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)

        return summarization

    def calculator_tool(self):
        @tool("calculator")
        def calculator(expression: str) -> str:
            """Evaluate a basic arithmetic expression, e.g. '12 * (5 + 3) / 2'.
            Only numbers and + - * / % ** operators are supported."""
            try:
                parsed = ast.parse(expression, mode="eval").body
                result = _safe_eval(parsed)
                return str(result)
            except Exception as exc:
                return f"Could not evaluate expression: {exc}"

        return calculator

    def csv_analysis_tool(self):
        documents_dir = settings.paths.documents_dir

        @tool("csv_analysis")
        def csv_analysis(file_name: str) -> str:
            """Analyze a CSV file from the documents folder and return summary
            statistics: shape, columns, dtypes, and basic numeric statistics.
            Provide the exact file name including the .csv extension."""
            file_path = os.path.join(documents_dir, file_name)
            if not os.path.isfile(file_path):
                return f"File not found: {file_name}"
            try:
                data_frame = pd.read_csv(file_path)
                description = data_frame.describe(include="all").to_string()
                return (
                    f"Shape: {data_frame.shape}\n"
                    f"Columns: {list(data_frame.columns)}\n"
                    f"Dtypes:\n{data_frame.dtypes.to_string()}\n\n"
                    f"Statistics:\n{description}"
                )
            except Exception as exc:
                return f"Failed to analyze CSV file: {exc}"

        return csv_analysis

    def excel_analysis_tool(self):
        documents_dir = settings.paths.documents_dir

        @tool("excel_analysis")
        def excel_analysis(file_name: str) -> str:
            """Analyze an Excel (.xlsx) file from the documents folder and
            return summary statistics for each sheet. Provide the exact file
            name including the .xlsx extension."""
            file_path = os.path.join(documents_dir, file_name)
            if not os.path.isfile(file_path):
                return f"File not found: {file_name}"
            try:
                sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
                parts = []
                for sheet_name, data_frame in sheets.items():
                    parts.append(
                        f"Sheet: {sheet_name}\n"
                        f"Shape: {data_frame.shape}\n"
                        f"Columns: {list(data_frame.columns)}\n"
                        f"Statistics:\n{data_frame.describe(include='all').to_string()}"
                    )
                return "\n\n".join(parts)
            except Exception as exc:
                return f"Failed to analyze Excel file: {exc}"

        return excel_analysis
