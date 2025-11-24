# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.embedder import Embedder
import faiss, json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE / "models" / "faiss.index"
META_PATH = BASE / "models" / "meta.jsonl"

app = FastAPI(title="Embedding Search API")
embedder = Embedder()
index = None
meta = []

@app.on_event("startup")
def load_index():
    global index, meta
    index = faiss.read_index(str(INDEX_PATH))
    meta = [json.loads(l) for l in open(META_PATH, encoding="utf-8")]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search(req: SearchRequest):
    qv = embedder.embed_query(req.query).astype("float32")
    faiss.normalize_L2(qv.reshape(1,-1))
    D,I = index.search(qv.reshape(1,-1), req.top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        m = meta[idx]
        results.append({"doc_id": m["doc_id"], "filename": m["filename"], "score": float(score)})
    return {"results": results}

# src/api.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# from src.embedder import Embedder
# from src.cache_manager import CacheManager
# from src.search_engine import SearchEngine
# import uvicorn
# import os

# app = FastAPI(title="Embedding Search API")

# # initialize components (in-memory singletons)
# embedder = Embedder()
# cache = CacheManager()
# search_engine = SearchEngine(embedder=embedder, cache=cache, dim=384)
# # load prebuilt index on startup if exists
# search_engine.load_index()

# class SearchRequest(BaseModel):
#     query: str
#     top_k: int = 5

# @app.post("/search")
# def search(req: SearchRequest):
#     if search_engine.index is None:
#         return {"error": "Index not built. Run build_index step first."}
#     results = search_engine.search(req.query, top_k=req.top_k)
#     return {"results": results}

# # For debug running directly:
# if __name__ == "__main__":
#     uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
