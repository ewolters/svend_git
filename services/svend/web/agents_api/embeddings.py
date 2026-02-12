"""Embedding service for semantic similarity search.

Uses sentence-transformers for local embedding generation
and pgvector for efficient similarity search in PostgreSQL.
"""

import logging
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, 384 dimensions
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the sentence-transformer model (cached singleton).

    Uses GPU if available, falls back to CPU.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model '{EMBEDDING_MODEL}' on {device}")

        model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        return model
    except ImportError:
        logger.error("sentence-transformers not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return None


def generate_embedding(text: str) -> Optional[np.ndarray]:
    """Generate embedding vector for text.

    Args:
        text: The text to embed

    Returns:
        numpy array of shape (384,) or None if failed
    """
    if not text or not text.strip():
        return None

    model = get_embedding_model()
    if model is None:
        return None

    try:
        # Truncate very long text (model has 256 token limit)
        text = text[:8000]  # Approximate character limit

        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return None


def generate_rca_embedding(event: str, chain: list, root_cause: str = "") -> Optional[np.ndarray]:
    """Generate embedding for an RCA session.

    Combines event description, causal chain claims, and root cause
    into a single embedding that captures the essence of the incident.

    Args:
        event: The incident description
        chain: List of causal chain steps with 'claim' keys
        root_cause: The identified root cause (if any)

    Returns:
        numpy array of shape (384,) or None if failed
    """
    parts = [event]

    # Add each causal claim from the chain
    for step in chain:
        claim = step.get("claim", "")
        if claim:
            parts.append(claim)

    # Add root cause if present
    if root_cause:
        parts.append(f"Root cause: {root_cause}")

    # Join with newlines for semantic separation
    combined_text = "\n".join(parts)

    return generate_embedding(combined_text)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def find_similar_in_memory(
    query_embedding: np.ndarray,
    embeddings: list[tuple[str, np.ndarray]],
    top_k: int = 5,
    threshold: float = 0.5,
) -> list[tuple[str, float]]:
    """Find similar items using in-memory search.

    Fallback for when pgvector isn't available.

    Args:
        query_embedding: The query vector
        embeddings: List of (id, embedding) tuples
        top_k: Number of results to return
        threshold: Minimum similarity score

    Returns:
        List of (id, similarity_score) tuples, sorted by score descending
    """
    scores = []

    for item_id, embedding in embeddings:
        if embedding is None:
            continue
        score = cosine_similarity(query_embedding, embedding)
        if score >= threshold:
            scores.append((item_id, score))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_k]
