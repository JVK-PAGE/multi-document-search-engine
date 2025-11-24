# Multi-Document Semantic Search Engine

A production-style **semantic search engine** built using:

* **Sentence-Transformers** (MiniLM-L6-v2)
* **FAISS vector index** (cosine similarity)
* **SQLite caching** for embeddings
* **FastAPI** backend

This project implements a complete retrieval pipeline from raw documents → cleaned data → embeddings → FAISS index → API search.

---

## 🚀 Features

* **Text Preprocessing**

  * Clean HTML, lowercase, remove noise
  * Compute SHA-256 hash for cache validation
* **Efficient Embedding Generator**

  * Uses MiniLM-L6-v2 (384-dim)
  * Batch processing
  * **Smart caching** (only re-embeds changed docs)
* **FAISS Vector Search Index**

  * IndexFlatIP (cosine similarity via L2 normalization)
  * Fast search over 10k+ documents
* **FastAPI REST API**

  * `/search` endpoint
  * Swagger UI documentation at `/docs`
* **Modular Architecture**

  * `embedder.py`, `cache_sqlite.py`, `search_engine.py`, `api.py`

---

## 📁 Project Structure

```
multi-document-search-engine/
├── src/
│   ├── preprocess.py
│   ├── embedder.py
│   ├── cache_sqlite.py
│   ├── generate_embeddings.py
│   ├── build_faiss_index.py
│   ├── search_engine.py
│   ├── api.py
│   └── save_20newsgroups.py
│
├── data/
│   ├── docs/              # raw documents
│   ├── processed/         # cleaned documents
│   └── metadata.json
│
├── models/
│   ├── embeddings_cache.sqlite
│   ├── faiss.index
│   └── meta.jsonl
│
├── requirements.txt
└── README.md
```

---

## 📘 1. Setup Instructions

### 1️⃣ Create & activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.\.venv\Scripts\activate   # Windows
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

(If you're on Windows and FAISS fails, install via Conda or use CPU-only PyTorch.)

---

## 📘 2. Download Dataset (20 Newsgroups)

This script downloads 11k documents and saves them as `.txt` files.

```bash
python -m src.save_20newsgroups
```

To sample only 150 docs, edit inside the script:

```python
SAMPLE_N = 150
```

---

## 📘 3. Preprocess Documents

```bash
python -m src.preprocess
```

This creates:

* cleaned files in `data/processed/`
* metadata.json

---

## 📘 4. Generate Embeddings (with caching)

```bash
python -m src.generate_embeddings
```

First run will embed all docs (slow).
Future runs detect unchanged files and skip embedding.

---

## 📘 5. Build FAISS Index

```bash
python -m src.build_faiss_index
```

Creates:

* `models/faiss.index`
* `models/meta.jsonl`

---

## 📘 6. Run the FastAPI Server

```bash
uvicorn src.api:app --reload --port 8000
```

API available at:

* Swagger UI: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
* Redoc: **[http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)**

### Example Query

```json
POST /search
{
  "query": "space exploration",
  "top_k": 5
}
```

---

## 🧠 How the System Works (Interview Explanation)

1. **Preprocess:** Normalize text and compute file hash.
2. **Embed:** MiniLM model converts text → 384-d embedding.
   Cache ensures we embed a document only if it has changed.
3. **Index:** FAISS stores embeddings and enables fast similarity search.
4. **Search API:** Embed query → FAISS search → return ranked results.

This architecture is the same used in real-world RAG systems.

---

## 🛠 Tech Stack

* Python 3
* Sentence Transformers
* FAISS
* FastAPI / Uvicorn
* SQLite
* NumPy / Scikit-learn

