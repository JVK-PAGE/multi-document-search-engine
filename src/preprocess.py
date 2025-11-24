# src/preprocess.py
from pathlib import Path
import hashlib
import json
import re
from tqdm import tqdm

# --- Robust path handling (project-root relative) ---
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
RAW_DIR = BASE_DIR / "data" / "docs"               # expected raw input
PROCESSED_DIR = BASE_DIR / "data" / "processed"    # cleaned output
METADATA_PATH = BASE_DIR / "data" / "metadata.json"  # metadata file

_html_tag_re = re.compile(r"<[^>]+>")
_multi_space_re = re.compile(r"\s+")

def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

def list_txt_files(folder: Path):
    return sorted(folder.glob("*.txt"))

def sha256_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def clean_text(text: str) -> str:
    text = _html_tag_re.sub(" ", text)
    text = text.lower()
    text = _multi_space_re.sub(" ", text).strip()
    return text

def process_one_file(src_path: Path):
    raw = src_path.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_text(raw)
    file_name = src_path.name
    doc_id = src_path.stem
    hash_val = sha256_text(cleaned)
    doc_length = len(cleaned)
    out_path = PROCESSED_DIR / f"{doc_id}.txt"
    out_path.write_text(cleaned, encoding="utf-8")
    meta = {
        "doc_id": doc_id,
        "filename": file_name,
        "clean_path": str(out_path),
        "doc_length": doc_length,
        "sha256": hash_val,
        "original_path": str(src_path),
        "preview": cleaned[:300]
    }
    return meta

def preprocess_documents():
    ensure_dirs()
    print(f"Project root: {BASE_DIR}")
    print(f"Looking for .txt files in: {RAW_DIR}")
    files = list_txt_files(RAW_DIR)
    print(f"Found {len(files)} text files.")
    metas = []
    if not files:
        # still create metadata file (empty list) so downstream scripts don't crash
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return
    with open(METADATA_PATH, "w", encoding="utf-8") as meta_f:
        for f in tqdm(files, desc="Processing Files"):
            meta = process_one_file(f)
            metas.append(meta)
            meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"Processed {len(metas)} files. Metadata saved to {METADATA_PATH}")

if __name__ == "__main__":
    preprocess_documents()

