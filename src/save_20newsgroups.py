# src/save_20newsgroups.py
import os
from sklearn.datasets import fetch_20newsgroups
from pathlib import Path
import random

OUTPUT_DIR = "data/docs"   # raw docs (preprocess expects data/docs -> data/processed)
SUBSET = "train"           # 'train' or 'test' or both
SAMPLE_N = None            # set to an int (e.g. 150) to limit to 100-200 files; set to None for all

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def sanitize_filename(s, max_len=80):
    # crude sanitize to make filename-safe doc ids
    import re
    s = re.sub(r"[^0-9a-zA-Z_\-]", "_", s)
    return s[:max_len]

def main():
    ensure_dir(OUTPUT_DIR)
    print(f"Downloading 20newsgroups subset='{SUBSET}' ... (this may take a minute)")
    data = fetch_20newsgroups(subset=SUBSET, remove=())  # you can pass remove=('headers','footers','quotes') if desired
    print("Downloaded. Total docs:", len(data.data))

    indices = list(range(len(data.data)))
    if SAMPLE_N is not None:
        random.seed(42)
        indices = random.sample(indices, min(SAMPLE_N, len(indices)))
        print(f"Sampling {len(indices)} docs (SAMPLE_N={SAMPLE_N})")

    for i, idx in enumerate(indices, start=1):
        text = data.data[idx]
        # build filename: doc_0001_<short-subject>.txt
        target_name = data.target_names[data.target[idx]] if hasattr(data, "target_names") else str(data.target[idx])
        fname = f"doc_{i:04d}_{sanitize_filename(target_name)}.txt"
        out_path = os.path.join(OUTPUT_DIR, fname)
        with open(out_path, "w", encoding="utf-8", errors="ignore") as fh:
            fh.write(text)
    print("Saved files to", OUTPUT_DIR)

if __name__ == "__main__":
    main()

