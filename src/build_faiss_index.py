# src/build_faiss_index.py
from pathlib import Path
import sqlite3
import numpy as np
import faiss
import json

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "models" / "embeddings_cache.sqlite"
INDEX_PATH = BASE / "models" / "faiss.index"
META_PATH = BASE / "models" / "meta.jsonl"

def load_cached_embeddings(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT doc_id, embedding, dim, hash, filename FROM cache")
    rows = c.fetchall()
    conn.close()
    docs = []
    for doc_id, emb_blob, dim, h, filename in rows:
        arr = np.frombuffer(emb_blob, dtype=np.float32)
        if arr.size != dim:
            # try to reshape safely
            # if dim is flat length, keep as 1D
            pass
        docs.append({"doc_id": doc_id, "embedding": arr.astype("float32"), "hash": h, "filename": filename})
    return docs

def build_index(docs, dim):
    # stack embeddings
    X = np.vstack([d["embedding"] for d in docs]).astype("float32")
    # normalize for cosine via inner-product
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, str(INDEX_PATH))
    # write meta
    with open(META_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            meta = {"doc_id": d["doc_id"], "filename": d["filename"], "hash": d["hash"]}
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"Saved FAISS index ({index.ntotal} vectors) -> {INDEX_PATH}")
    print(f"Saved metadata -> {META_PATH}")

if __name__ == "__main__":
    docs = load_cached_embeddings(DB_PATH)
    if not docs:
        print("No cached embeddings found. Run generate_embeddings first.")
        raise SystemExit(1)
    dim = docs[0]["embedding"].shape[0]
    build_index(docs, dim)
