# src/build_index.py
import os
from glob import glob
from src.embedder import Embedder
from src.cache_manager import CacheManager
from src.search_engine import SearchEngine
import numpy as np

def load_text_files(folder="data"):
    files = sorted(glob(os.path.join(folder, "*.txt")))
    docs = []
    for i, f in enumerate(files):
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        doc_id = os.path.splitext(os.path.basename(f))[0]
        docs.append({"doc_id": doc_id, "filename": os.path.basename(f), "text": text})
    return docs

def main():
    embedder = Embedder()
    cache = CacheManager()
    search = SearchEngine(embedder, cache, dim=embedder.model.get_sentence_embedding_dimension())
    docs = load_text_files("data")
    # compute embeddings, use cache
    processed = []
    texts = [d['text'] for d in docs]
    hashes = [Embedder.sha256(t) for t in texts]

    # batch embed all texts (you could check cache per-file to skip)
    embeddings = embedder.embed_texts(texts, batch_size=32)
    for d, emb, h in zip(docs, embeddings, hashes):
        cache.upsert(d['doc_id'], emb, h, d['filename'])
        processed.append({"doc_id": d['doc_id'], "filename": d['filename'], "text": d['text'], "embedding": emb})

    search.build_index(processed)
    print("Index built and saved.")

if __name__ == "__main__":
    main()
