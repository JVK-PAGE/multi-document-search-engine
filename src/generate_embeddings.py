# src/generate_embeddings.py
import os
import glob
import hashlib
import json
from pathlib import Path
from tqdm import tqdm

from src.embedder import Embedder
from src.cache_manager import SQLiteCache

# from cache_manager import SQLiteCache  # recommended
# from src.cache_json import JSONCache  # alternative

PROCESSED_DIR = "data/processed"  # input
MANIFEST_PATH = "models/embedding_manifest.jsonl"  # record of what we processed

def sha256_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def list_processed_files(folder=PROCESSED_DIR):
    return sorted(glob.glob(os.path.join(folder, "*.txt")))

def load_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()

def save_manifest(records, path=MANIFEST_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

def main(use_json_cache=False):
    embedder = Embedder()
    cache = None
    if use_json_cache:
        from src.cache_json import JSONCache
        cache = JSONCache()
    else:
        cache = SQLiteCache()
    files = list_processed_files()
    print(f"Found {len(files)} processed files.")
    manifest = []
    # We'll collect texts to batch-embed only when needed
    to_embed = []  # list of dicts: {doc_id, filename, text}
    for p in files:
        text = load_text(p)
        doc_id = Path(p).stem
        filename = Path(p).name
        h = sha256_text(text)
        cached = cache.get(doc_id)
        if cached and cached["hash"] == h:
            # use cached embedding
            manifest.append({
                "doc_id": doc_id,
                "filename": filename,
                "sha256": h,
                "cached": True,
                "cached_updated_at": cached.get("updated_at")
            })
            continue
        # else schedule to embed
        to_embed.append({"doc_id": doc_id, "filename": filename, "text": text, "sha256": h})
    # batch embed
    if to_embed:
        texts = [d["text"] for d in to_embed]
        print(f"Embedding {len(texts)} documents ...")
        embeddings = embedder.embed_texts(texts, batch_size=32)
        for d, emb in zip(to_embed, embeddings):
            cache.upsert(d["doc_id"], emb, d["sha256"], d["filename"])
            manifest.append({
                "doc_id": d["doc_id"],
                "filename": d["filename"],
                "sha256": d["sha256"],
                "cached": False,
                "updated_at": None
            })
    # sort manifest for readability
    save_manifest(manifest)
    print("Done. Manifest written to", MANIFEST_PATH)
    # Print summary
    total = len(files)
    reused = sum(1 for m in manifest if m.get("cached"))
    newly = total - reused
    print(f"Total files: {total}, reused: {reused}, newly embedded: {newly}")

if __name__ == "__main__":
    main(use_json_cache=False)
