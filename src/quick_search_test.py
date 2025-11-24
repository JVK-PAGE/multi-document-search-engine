# src/quick_search_test.py
from pathlib import Path
import faiss, numpy as np, json
from src.embedder import Embedder

BASE = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE / "models" / "faiss.index"
META_PATH = BASE / "models" / "meta.jsonl"

index = faiss.read_index(str(INDEX_PATH))
meta = [json.loads(l) for l in open(META_PATH, encoding="utf-8")]
print("Index loaded. n_vectors:", index.ntotal)

embedder = Embedder()
q = "machine learning basics"
qv = embedder.embed_query(q).astype("float32")
faiss.normalize_L2(qv.reshape(1,-1))
D,I = index.search(qv.reshape(1,-1), 5)
for score, idx in zip(D[0], I[0]):
    print(score, meta[idx]["doc_id"], meta[idx]["filename"])
