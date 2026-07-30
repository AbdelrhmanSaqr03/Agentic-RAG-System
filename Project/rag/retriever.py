"""
Retrieval layer.
Provides a high-level interface for fetching relevant document chunks
from the FAISS vector store, including score-based filtering.
"""

from typing import List, Dict

from config import settings
from rag.vector_database import VectorDatabase
from utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """High-level retrieval interface built on top of the vector database."""

    def __init__(self, vector_database: VectorDatabase):
        self.vector_database = vector_database

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve the most relevant document chunks for a query.

        Args:
            query: The user query or rewritten search query.
            top_k: Number of chunks to retrieve.

        Returns:
            A list of dicts containing "content", "source", and "score".
        """
        if self.vector_database.store is None:
            logger.warning("Vector store is empty; no documents to retrieve from.")
            return []

        top_k = top_k or settings.retrieval.top_k

        try:
            results = self.vector_database.store.similarity_search_with_score(query, k=top_k)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return []

        formatted = []
        for document, score in results:
            formatted.append({
                "content": document.page_content,
                "source": document.metadata.get("source", "unknown"),
                "chunk_index": document.metadata.get("chunk_index", -1),
                "score": float(score),
            })
        return formatted

    def is_ready(self) -> bool:
        """Return True if the underlying vector store has been initialized."""
        return self.vector_database.store is not None
