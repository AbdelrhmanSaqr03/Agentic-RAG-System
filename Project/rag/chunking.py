"""
Text chunking layer.
Splits cleaned document text into semantically coherent chunks suitable
for embedding and retrieval, while preserving source metadata.
"""

from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """Splits documents into overlapping semantic chunks."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: List[Dict]) -> List[Document]:
        """
        Convert raw document dicts into a list of LangChain Document chunks.

        Args:
            documents: List of dicts with "source" and "text" keys.

        Returns:
            A list of langchain.schema.Document objects with metadata.
        """
        all_chunks: List[Document] = []

        for doc in documents:
            source_name = doc.get("source", "unknown")
            text = doc.get("text", "")

            if not text.strip():
                continue

            raw_chunks = self.splitter.split_text(text)
            for index, chunk_text in enumerate(raw_chunks):
                metadata = {
                    "source": source_name,
                    "chunk_index": index,
                }
                all_chunks.append(Document(page_content=chunk_text, metadata=metadata))

        logger.info("Created %d chunks from %d documents.", len(all_chunks), len(documents))
        return all_chunks
