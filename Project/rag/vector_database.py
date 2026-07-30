"""
Vector database layer.
Manages building, persisting, and loading the FAISS vector store that
backs the knowledge base.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import settings
from rag.embeddings import EmbeddingManager
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorDatabase:
    """Handles creation, persistence, and loading of the FAISS index."""

    def __init__(self):
        self.embedding_model = EmbeddingManager().get_model()
        self.index_path = os.path.join(
            settings.paths.vector_store_dir, settings.faiss_index_name
        )
        self.store: Optional[FAISS] = None

    def build_from_documents(self, documents: List[Document]) -> FAISS:
        """
        Build a new FAISS index from a list of chunked documents.

        Args:
            documents: List of Document chunks to embed and index.

        Returns:
            The constructed FAISS vector store.
        """
        if not documents:
            raise ValueError("Cannot build a vector database from an empty document list.")

        logger.info("Building FAISS index from %d chunks.", len(documents))
        self.store = FAISS.from_documents(documents, self.embedding_model)
        self._save()
        return self.store

    def load(self) -> Optional[FAISS]:
        """
        Load an existing FAISS index from disk if it exists.

        Returns:
            The loaded FAISS vector store, or None if no index is found.
        """
        if not self._index_exists():
            logger.info("No existing FAISS index found at %s", self.index_path)
            return None

        logger.info("Loading FAISS index from %s", self.index_path)
        self.store = FAISS.load_local(
            self.index_path,
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )
        return self.store

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new document chunks to an existing index, creating one if needed.

        Args:
            documents: List of Document chunks to add.
        """
        if not documents:
            return

        if self.store is None:
            self.build_from_documents(documents)
            return

        self.store.add_documents(documents)
        self._save()

    def _save(self) -> None:
        """Persist the current FAISS store to disk."""
        if self.store is not None:
            self.store.save_local(self.index_path)
            logger.info("FAISS index saved to %s", self.index_path)

    def _index_exists(self) -> bool:
        """Check whether a persisted FAISS index exists on disk."""
        return os.path.isdir(self.index_path) and bool(os.listdir(self.index_path)) \
            if os.path.exists(self.index_path) else False
