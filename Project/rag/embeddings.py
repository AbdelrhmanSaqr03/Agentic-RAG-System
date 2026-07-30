"""
Embeddings layer.
Wraps the Sentence Transformers model used to embed document chunks
and user queries for vector similarity search.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingManager:
    """Provides a singleton-style access point to the embedding model."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def get_model(self) -> HuggingFaceEmbeddings:
        """
        Lazily instantiate and return the embedding model.

        Returns:
            A configured HuggingFaceEmbeddings instance.
        """
        if self._model is None:
            logger.info("Loading embedding model: %s", settings.model.embedding_model)
            self._model = HuggingFaceEmbeddings(
                model_name=settings.model.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._model
