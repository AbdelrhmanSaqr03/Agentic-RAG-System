"""
Central configuration module for the Agentic RAG System.
Loads environment variables and exposes typed configuration objects
used across the entire application.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Paths:
    """Filesystem paths used throughout the project."""
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    documents_dir: str = os.path.join(base_dir, "documents")
    vector_store_dir: str = os.path.join(base_dir, "vector_store")
    data_dir: str = os.path.join(base_dir, "data")
    logs_dir: str = os.path.join(base_dir, "data", "logs")


@dataclass
class ModelConfig:
    """LLM and embedding model configuration."""
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    gemini_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class ChunkingConfig:
    """Text splitting configuration."""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))


@dataclass
class RetrievalConfig:
    """Retriever behavior configuration."""
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    score_threshold: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.0"))


@dataclass
class AppConfig:
    """Aggregated application configuration."""
    paths: Paths = field(default_factory=Paths)
    model: ModelConfig = field(default_factory=ModelConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    supported_extensions: tuple = (".pdf", ".docx", ".txt", ".csv", ".xlsx")
    faiss_index_name: str = "faiss_index"
    app_title: str = "Agentic RAG Knowledge Assistant"


settings = AppConfig()

os.makedirs(settings.paths.documents_dir, exist_ok=True)
os.makedirs(settings.paths.vector_store_dir, exist_ok=True)
os.makedirs(settings.paths.data_dir, exist_ok=True)
os.makedirs(settings.paths.logs_dir, exist_ok=True)
