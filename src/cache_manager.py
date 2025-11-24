# src/cache_manager.py
import sqlite3
import numpy as np
import time
import os
from typing import Optional, Dict
DB_PATH = "models/embeddings_cache.sqlite"

def ensure_db(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS cache (
        doc_id TEXT PRIMARY KEY,
        embedding BLOB,
        dim INTEGER,
        hash TEXT,
        updated_at REAL,
        filename TEXT
    )
    """)
    conn.commit()
    conn.close()

def np_to_blob(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def blob_to_np(blob: bytes, dim: int, dtype=np.float32) -> np.ndarray:
    a = np.frombuffer(blob, dtype=dtype)
    if a.size != dim:
        # try to reshape safely if possible
        return a.reshape(-1)  # leave it 1D; caller can reshape
    return a

class SQLiteCache:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        ensure_db(self.db_path)

    def get(self, doc_id: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT embedding, dim, hash, updated_at, filename FROM cache WHERE doc_id = ?", (doc_id,))
        row = c.fetchone()
        conn.close()
        if row is None:
            return None
        embedding_blob, dim, hash_val, updated_at, filename = row
        return {
            "embedding_blob": embedding_blob,
            "dim": dim,
            "hash": hash_val,
            "updated_at": updated_at,
            "filename": filename
        }

    def upsert(self, doc_id: str, embedding: np.ndarray, hash_val: str, filename: str):
        emb_blob = np_to_blob(embedding.astype("float32"))
        dim = embedding.size
        ts = time.time()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        INSERT INTO cache(doc_id, embedding, dim, hash, updated_at, filename)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
           embedding = excluded.embedding,
           dim = excluded.dim,
           hash = excluded.hash,
           updated_at = excluded.updated_at,
           filename = excluded.filename
        """, (doc_id, emb_blob, dim, hash_val, ts, filename))
        conn.commit()
        conn.close()

    def list_all(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT doc_id, dim, hash, updated_at, filename FROM cache")
        rows = c.fetchall()
        conn.close()
        return rows
