# src/search_engine.py
import faiss
import numpy as np
import os
from typing import List, Dict
from src.embedder import Embedder
from src.cache_manager import CacheManager, blob_to_np

INDEX_PATH = "models/faiss.index"
META_PATH = "models/meta.npy"  # store doc ids and metadata

class SearchEngine:
    def __init__(self, embedder: Embedder, cache: CacheManager, dim: int = 384):
        self.embedder = embedder
        self.cache = cache
        self.dim = dim
        self.index = None
        self.docid_map = []  # list of dicts: {doc_id, filename, preview, length}
        os.makedirs("models", exist_ok=True)

    def build_index(self, docs: List[Dict]):
        """
        docs: list of {'doc_id','filename','text','embedding'}
        """
        embeddings = np.vstack([d['embedding'] for d in docs]).astype('float32')
        # normalize for cosine (optional)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.docid_map = [{'doc_id': d['doc_id'], 'filename': d['filename'], 'preview': d['text'][:200], 'length': len(d['text'])} for d in docs]
        faiss.write_index(self.index, INDEX_PATH)
        np.save(META_PATH, self.docid_map, allow_pickle=True)

    def load_index(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            self.docid_map = np.load(META_PATH, allow_pickle=True).tolist()

    def search(self, query: str, top_k: int = 5):
        q_emb = self.embedder.embed_query(query).astype('float32')
        faiss.normalize_L2(q_emb.reshape(1,-1))
        D, I = self.index.search(q_emb.reshape(1,-1), top_k)
        results = []
        sim_scores = D[0].tolist()
        ids = I[0].tolist()
        for score, idx in zip(sim_scores, ids):
            meta = self.docid_map[idx]
            explanation = self._explain(query, meta['filename'], meta['preview'])
            results.append({
                "doc_id": meta['doc_id'],
                "score": float(score),
                "preview": meta['preview'],
                "explanation": explanation
            })
        return results

    def _explain(self, query: str, filename: str, preview: str):
        # Simple keyword overlap heuristic
        query_tokens = set(query.lower().split())
        doc_tokens = set(preview.lower().split())
        overlap = query_tokens & doc_tokens
        overlap_ratio = len(overlap) / (len(query_tokens) + 1e-9)
        return {
            "matched_keywords": list(overlap)[:10],
            "overlap_ratio": overlap_ratio,
            "note": f"Preview from {filename}"
        }
