"""Embedding service using Sentence Transformers (Free Local Model)"""

from typing import List, Optional
import numpy as np

class EmbeddingService:
    """
    Free local embedding service using sentence-transformers.
    Uses 'all-MiniLM-L6-v2' - a fast, efficient model with 384 dimensions.
    No API key needed, runs completely locally.
    """

    def __init__(self, api_key: str = None, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.
        api_key is kept for backward compatibility but not used.
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model with GPU support"""
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            # Check for GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading embedding model: {self.model_name} on {device.upper()}...")

            self.model = SentenceTransformer(self.model_name, device=device)
            print(f"Embedding model loaded successfully on {device.upper()}! Dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.model = None

    def get_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Get embedding for a single text"""
        try:
            if self.model is None:
                print("Model not loaded, returning zero vector")
                return [0.0] * self.embedding_dim

            # Clean the text
            text = text.strip()
            if not text:
                return [0.0] * self.embedding_dim

            # Get embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0] * self.embedding_dim

    def get_embeddings_batch(self, texts: List[str],
                             task_type: str = "retrieval_document",
                             batch_size: int = 32) -> List[List[float]]:
        """Get embeddings for multiple texts with batching"""
        try:
            if self.model is None:
                print("Model not loaded, returning zero vectors")
                return [[0.0] * self.embedding_dim] * len(texts)

            # Clean texts
            cleaned_texts = [t.strip() if t else "" for t in texts]

            # Get all embeddings in batch (much faster than one by one)
            print(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.model.encode(
                cleaned_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            print(f"Generated {len(embeddings)} embeddings successfully!")
            return [emb.tolist() for emb in embeddings]

        except Exception as e:
            print(f"Error processing batch: {e}")
            return [[0.0] * self.embedding_dim] * len(texts)

    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a search query"""
        return self.get_embedding(query, task_type="retrieval_query")

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # Convert to numpy for efficient computation
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 * norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))

        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0


# Alternative model options (can be changed in __init__):
# - "all-MiniLM-L6-v2": Fast, 384 dims, good quality (DEFAULT)
# - "all-mpnet-base-v2": Best quality, 768 dims, slower
# - "paraphrase-MiniLM-L6-v2": Good for paraphrase detection
# - "multi-qa-MiniLM-L6-cos-v1": Optimized for Q&A
