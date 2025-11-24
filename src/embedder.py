# src/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
from typing import List
from tqdm import tqdm


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        # device can be "cuda" or "cpu" or None (auto)
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Return shape (N, dim) float32 numpy array
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        embs = self.model.encode(texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True)
        return embs.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        emb = self.model.encode([text], convert_to_numpy=True)[0]
        return emb.astype("float32")

