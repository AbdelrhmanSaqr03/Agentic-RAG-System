"""
Core agent orchestration layer.
Defines the LangGraph workflow that plans, retrieves, uses tools, and
generates cited answers for the Agentic RAG system.
"""

import json
import re
from typing import TypedDict, List, Dict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from config import settings
from agent.prompts import PLANNER_PROMPT, ANSWER_PROMPT
from agent.memory import ConversationMemory
from agent.tools import ToolFactory
from rag.retriever import Retriever
from utils.constants import DEFAULT_ANSWER_NO_CONTEXT
from utils.helpers import format_sources
from utils.logger import get_logger

logger = get_logger(__name__)


class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""
    question: str
    history: str
    plan: Dict
    search_query: str
    needs_retrieval: bool
    requested_tools: List[str]
    retrieved_chunks: List[Dict]
    tool_outputs: str
    answer: str
    sources: List[Dict]


class AgenticRAG:
    """Builds and runs the LangGraph-based agentic RAG workflow."""

    def __init__(self, retriever: Retriever, memory: Optional[ConversationMemory] = None):
        self.retriever = retriever
        self.memory = memory or ConversationMemory()
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model.gemini_model,
            google_api_key=settings.model.google_api_key,
            temperature=settings.model.gemini_temperature,
        )
        self.tools = {t.name: t for t in ToolFactory(retriever, self.llm).build_all()}
        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct the LangGraph state machine for the agent workflow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("plan", self._plan_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("use_tools", self._tools_node)
        workflow.add_node("answer", self._answer_node)

        workflow.set_entry_point("plan")

        workflow.add_conditional_edges(
            "plan",
            lambda state: "retrieve" if state.get("needs_retrieval") else "use_tools",
            {"retrieve": "retrieve", "use_tools": "use_tools"},
        )
        workflow.add_edge("retrieve", "use_tools")
        workflow.add_edge("use_tools", "answer")
        workflow.add_edge("answer", END)

        return workflow.compile()

    def _plan_node(self, state: AgentState) -> AgentState:
        """Decide whether retrieval and/or tools are needed, and rewrite the query."""
        prompt = PLANNER_PROMPT.format(history=state["history"], question=state["question"])
        try:
            response = self.llm.invoke(prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)
            plan = self._parse_plan(raw_text, state["question"])
        except Exception as exc:
            logger.error("Planning step failed: %s", exc)
            plan = {
                "needs_retrieval": True,
                "tools": [],
                "search_query": state["question"],
            }

        state["plan"] = plan
        state["needs_retrieval"] = plan.get("needs_retrieval", True)
        state["requested_tools"] = plan.get("tools", [])
        state["search_query"] = plan.get("search_query", state["question"])
        return state

    @staticmethod
    def _parse_plan(raw_text: str, fallback_question: str) -> Dict:
        """Extract a JSON plan object from the LLM's raw text response."""
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            return {
                "needs_retrieval": True,
                "tools": [],
                "search_query": fallback_question,
            }
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "needs_retrieval": True,
                "tools": [],
                "search_query": fallback_question,
            }

    def _retrieve_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant chunks from the vector store using the rewritten query."""
        chunks = self.retriever.retrieve(state["search_query"])
        state["retrieved_chunks"] = chunks
        state["sources"] = [
            {"source": c["source"], "score": c["score"]} for c in chunks
        ]
        return state

    def _tools_node(self, state: AgentState) -> AgentState:
        """Invoke any tools requested by the planner (excluding document_search)."""
        requested = [t for t in state.get("requested_tools", []) if t != "document_search"]
        outputs = []

        for tool_name in requested:
            tool_fn = self.tools.get(tool_name)
            if tool_fn is None:
                continue
            try:
                result = tool_fn.invoke({
                    "expression": state["question"],
                    "text": state["question"],
                    "file_name": self._guess_file_name(state["question"]),
                    "query": state["search_query"],
                }.get(self._primary_arg_name(tool_name), state["question"]))
                outputs.append(f"[{tool_name}] {result}")
            except Exception as exc:
                logger.warning("Tool '%s' failed: %s", tool_name, exc)

        state["tool_outputs"] = "\n".join(outputs) if outputs else "No additional tools used."
        return state

    @staticmethod
    def _primary_arg_name(tool_name: str) -> str:
        """Map a tool name to the argument key used to invoke it."""
        mapping = {
            "calculator": "expression",
            "summarization": "text",
            "csv_analysis": "file_name",
            "excel_analysis": "file_name",
        }
        return mapping.get(tool_name, "query")

    @staticmethod
    def _guess_file_name(question: str) -> str:
        """Best-effort extraction of a referenced file name from the question."""
        match = re.search(r"[\w\-. ]+\.(csv|xlsx)", question, re.IGNORECASE)
        return match.group(0) if match else ""

    def _answer_node(self, state: AgentState) -> AgentState:
        """Generate the final cited answer using retrieved context and tool outputs."""
        chunks = state.get("retrieved_chunks", [])
        context = "\n---\n".join(
            f"(Source: {c['source']}) {c['content']}" for c in chunks
        ) if chunks else "No retrieved context."

        prompt = ANSWER_PROMPT.format(
            history=state["history"],
            context=context,
            tool_outputs=state.get("tool_outputs", "None."),
            question=state["question"],
        )

        try:
            response = self.llm.invoke(prompt)
            answer_text = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.error("Answer generation failed: %s", exc)
            answer_text = DEFAULT_ANSWER_NO_CONTEXT

        state["answer"] = answer_text
        return state

    def ask(self, question: str) -> Dict:
        """
        Run the full agentic workflow for a single user question.

        Args:
            question: The user's natural language question.

        Returns:
            A dict with "answer", "sources", and "search_query" keys.
        """
        initial_state: AgentState = {
            "question": question,
            "history": self.memory.get_history_text(),
        }

        final_state = self.graph.invoke(initial_state)
        answer = final_state.get("answer", DEFAULT_ANSWER_NO_CONTEXT)
        sources = final_state.get("sources", [])

        self.memory.add_turn(question, answer)

        return {
            "answer": answer,
            "sources": sources,
            "sources_text": format_sources(sources),
            "search_query": final_state.get("search_query", question),
        }
