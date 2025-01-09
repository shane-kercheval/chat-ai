"""VectorDB class for document chunking and search."""
from dataclasses import dataclass
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from cachetools import LRUCache

@dataclass
class Document:
    """Document to be processed."""

    key: str
    content: str
    last_modified: str


@dataclass
class Chunk:
    """Represents a chunk of text with its embedding."""

    key: str
    text: str  # The main chunk text
    index: int
    embedding: np.ndarray | None = None  # Optional as it's added later


@dataclass
class SearchResult:
    """Search result containing chunk info without embedding."""

    key: str
    text: str
    index: int
    score: float


def chunk_text(text: str, key: str, chunk_size: int = 500) -> list[Chunk]:
    """
    Split text into non-overlapping chunks.

    Args:
        text: Text to split
        key: Document key for chunks
        chunk_size: Target size of each chunk in characters

    Returns:
        List of Chunk objects (without embeddings)
    """
    if not text:
        return []

    # Normalize whitespace
    text = ' '.join(text.split())
    text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_sentences = []
    current_size = 0
    for sentence in sentences:
        sentence_len = len(sentence)
        if current_size + sentence_len > chunk_size and current_sentences:
            # Create chunk from current sentences
            chunks.append(Chunk(
                key=key,
                text=' '.join(current_sentences),
                index=len(chunks),
            ))
            # Reset for next chunk
            current_sentences = []
            current_size = 0
        current_sentences.append(sentence)
        current_size += sentence_len

    # Add final chunk if there are remaining sentences
    if current_sentences:
        chunks.append(Chunk(
            key=key,
            text=' '.join(current_sentences),
            index=len(chunks),
        ))

    return chunks


class SimilarityScorer:
    """In memory vector database for document chunking and search."""

    def __init__(
            self,
            model: SentenceTransformer,
            cache_size: int = 50,
            chunk_size: int = 500,
        ):
        """
        Score chunks from Documents against a query using cosine similarity.

        Args:
            model: SentenceTransformer model for creating embeddings
            cache_size: Number of documents (and corresponding embeddings) to cache in memory
            chunk_size: Target size of each chunk in characters. Chunks retain sentence boundaries.
        """
        self.model = model
        self._cache = LRUCache(maxsize=cache_size)
        self.chunk_size = chunk_size

    def _get_cache_key(self, doc: Document) -> str:
        """Get key for document cache."""
        return f"{doc.key}_{doc.last_modified}"

    def _get_document_chunks(self, documents: list[Document]) -> list[Chunk]:
        """Get or compute chunks for all documents."""
        all_chunks = []
        texts_to_embed = []
        chunks_without_embeddings = []  # Track chunks that need embeddings

        # First pass: get cached chunks and collect texts that need embedding
        for doc in documents:
            cached_chunks = self._cache.get(self._get_cache_key(doc))
            if cached_chunks:
                all_chunks.extend(cached_chunks)
            else:
                doc_chunks = chunk_text(doc.content, doc.key, self.chunk_size)
                all_chunks.extend(doc_chunks)
                for chunk in doc_chunks:
                    texts_to_embed.append(chunk.text)
                    chunks_without_embeddings.append(chunk)

        # Compute embeddings for all new chunks
        if texts_to_embed:
            embeddings = self.model.encode(texts_to_embed)
            # Update chunks with embeddings
            for chunk, embedding in zip(chunks_without_embeddings, embeddings, strict=True):
                chunk.embedding = embedding

            # Update cache
            chunks_by_key = {}
            for chunk in chunks_without_embeddings:
                if chunk.key not in chunks_by_key:
                    chunks_by_key[chunk.key] = []
                chunks_by_key[chunk.key].append(chunk)
            for doc in documents:
                doc_chunks = chunks_by_key.get(doc.key)
                if doc_chunks:
                    self._cache[self._get_cache_key(doc)] = doc_chunks

        return all_chunks

    def _cosine_similarity_matrix(
            self,
            query_embedding: np.ndarray,
            chunk_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all chunks at once."""
        # Normalize the embeddings
        query_norm = np.linalg.norm(query_embedding)
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
        # Avoid division by zero
        query_norm = np.maximum(query_norm, 1e-10)
        chunk_norms = np.maximum(chunk_norms, 1e-10)
        # Calculate similarities
        return np.dot(chunk_embeddings, query_embedding) / (chunk_norms * query_norm)

    def score(self, documents: list[Document], query: str) -> list[SearchResult]:
        """Score all chunks against query."""
        # Get all chunks with embeddings
        chunks = self._get_document_chunks(documents)
        if not chunks:
            return []
        # Encode query
        query_embedding = self.model.encode(query)
        # Get all chunk embeddings as a matrix
        chunk_embeddings = np.vstack([chunk.embedding for chunk in chunks])
        # Calculate similarities for all chunks at once
        similarities = self._cosine_similarity_matrix(query_embedding, chunk_embeddings)
        # Convert to SearchResults
        return [
            SearchResult(
                key=chunk.key,
                text=chunk.text,
                index=chunk.index,
                score=float(score),  # Convert from numpy float to Python float
            )
            for chunk, score in zip(chunks, similarities)
        ]
