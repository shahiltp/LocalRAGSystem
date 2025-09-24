import os
import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from llama_index.core import SimpleDirectoryReader
from docling.document_converter import DocumentConverter

# Input and output directories
input_dir = Path("/Users/kiwitech/Documents/agentic-rag-poc/data/raw")
output_root = Path("/Users/kiwitech/Documents/agentic-rag-poc/data/processed")
json_dir = output_root / "json"
md_dir = output_root / "md"
json_dir.mkdir(parents=True, exist_ok=True)
md_dir.mkdir(parents=True, exist_ok=True)

# --- Use LlamaIndex to load "documents" (kept per your request)
reader = SimpleDirectoryReader(input_dir=str(input_dir))
documents = reader.load_data()

def _extract_source_path(doc):
    """Best-effort: pull original file path from LlamaIndex metadata."""
    meta = getattr(doc, "metadata", {}) or {}
    for k in ("file_path", "filepath", "path", "source", "filename", "file_name"):
        v = meta.get(k)
        if not v:
            continue
        p = Path(v)
        if not p.is_absolute():
            p = input_dir / v
        if p.exists():
            return p.resolve()
    # Fallback: doc_id as path (rare)
    try:
        p = Path(getattr(doc, "doc_id", ""))
        if p.exists():
            return p.resolve()
    except Exception:
        pass
    return None

# Gather candidate paths from LlamaIndex docs
candidate_paths = []
for d in documents:
    p = _extract_source_path(d)
    if p and p.is_file():
        candidate_paths.append(p)

# If nothing came from LI metadata, fallback to directory listing
if not candidate_paths:
    candidate_paths = [p.resolve() for p in input_dir.iterdir() if p.is_file()]

# De-duplicate while preserving order
seen = set()
unique_paths = []
dupe_counts = Counter()
for p in candidate_paths:
    if p in seen:
        dupe_counts[p.name] += 1
        continue
    seen.add(p)
    unique_paths.append(p)

if dupe_counts:
    print(f"ℹ️ De-duplicated {sum(dupe_counts.values())} extra references across {len(dupe_counts)} files.")

# --- Convert each unique file exactly once
converter = DocumentConverter()

for src_path in tqdm(unique_paths, desc="Converting", unit="file", total=len(unique_paths)):
    base_name = src_path.stem
    out_json = json_dir / f"{base_name}.json"
    out_md = md_dir / f"{base_name}.md"
    try:
        conv_res = converter.convert(str(src_path))
        dl_doc = conv_res.document

        # JSON
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(dl_doc.export_to_dict(), f, ensure_ascii=False, indent=2)
        # Markdown
        out_md.write_text(dl_doc.export_to_markdown(), encoding="utf-8")

        print(f"✅ Saved: {out_json}")
        print(f"✅ Saved: {out_md}")
    except Exception as e:
        print(f"⚠️ Error processing {src_path.name}: {e}")
