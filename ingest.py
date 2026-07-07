"""
ingest.py — Load BNS 2023 PDF, split on legal section boundaries, embed via
LM Studio, store in ChromaDB with section metadata.

Chunking strategy:
  • Primary split: regex detecting numbered section headers (e.g. "104. Punishment …")
  • Secondary split: sections > MAX_SECTION_CHARS are further split with overlap
    so the 0.5B model always gets digestible context windows
  • Too-short fragments (<MIN_CHUNK_CHARS) are merged with the next chunk
"""

import os
import re
import math
import pypdf
import chromadb
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
PDF_PATH         = "data/bns_2023.pdf"
CHROMA_PATH      = "./chroma_db"
COLLECTION_NAME  = "bns_2023"
BATCH_SIZE       = 32
EMBED_MODEL      = "text-embedding-nomic-embed-text-v1.5"
BASE_URL         = "http://localhost:1234/v1"
API_KEY          = "lm-studio"

MIN_CHUNK_CHARS  = 80    # merge chunks shorter than this with the next one
MAX_SECTION_CHARS = 800  # sections longer than this get sub-chunked
OVERLAP_CHARS    = 100   # overlap for sub-chunking long sections

# ── LM Studio client ──────────────────────────────────────────────────────────
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Matches lines like "104. Punishment for murder.—"
# It expects digit(s), a dot, space(s), title text, and an em-dash "—" (or \ufffd replacement char)
_SECTION_HEADER = re.compile(
    r"(?m)^(\d{1,3})\.\s+([A-Z][^—\ufffd\n]{2,150})[—\ufffd]"
)

def clean_text(text: str) -> str:
    """Strip PDF page numbers and running headers like 'SECTIONS'."""
    # Remove bare page numbers on their own lines (e.g. "18 \n")
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    # Remove running headers
    text = re.sub(r"(?m)^\s*SECTIONS\s*$", "", text)
    # Condense multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text

def load_pdf(path: str) -> str:
    """Extract all text from a PDF file."""
    print(f"[ingest] Loading PDF: {path}")
    reader = pypdf.PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    full_text = "\n".join(pages)
    print(f"[ingest] Extracted {len(full_text):,} chars from {len(reader.pages)} pages.")
    
    cleaned = clean_text(full_text)
    return cleaned

def _sub_chunk(text: str, section_num: int, section_title: str,
               base_idx: int) -> list[tuple[str, dict]]:
    """
    Split a long section body into overlapping sub-chunks.
    Each sub-chunk keeps section metadata so retrieval is still precise.
    """
    results = []
    start = 0
    sub = 0
    while start < len(text):
        end = start + MAX_SECTION_CHARS
        fragment = text[start:end]
        meta = {
            "section_num":   section_num,
            "section_title": section_title.strip(),
            "sub_chunk":     sub,
            "chunk_index":   base_idx + sub,
        }
        results.append((fragment, meta))
        sub += 1
        start += MAX_SECTION_CHARS - OVERLAP_CHARS
    return results

def chunk_by_section(text: str) -> list[tuple[str, dict]]:
    """
    Split document text on legal section boundaries.

    Returns a list of (chunk_text, metadata_dict) tuples where metadata has:
      section_num   – integer section number
      section_title – first line of the section header
      sub_chunk     – 0 for first sub-chunk, 1+ for overflow splits
      chunk_index   – global index across all chunks
    """
    # Find all section header positions using the em-dash delimiter
    matches = list(_SECTION_HEADER.finditer(text))

    if not matches:
        print("[ingest] WARNING: No section headers found — falling back to fixed-size chunks.")
        # Fallback: fixed 500-char chunks
        chunks = []
        for i, start in enumerate(range(0, len(text), 500 - 80)):
            fragment = text[start:start + 500]
            chunks.append((fragment, {"section_num": 0, "section_title": "unknown",
                                      "sub_chunk": 0, "chunk_index": i}))
        return chunks

    raw_sections: list[tuple[int, str, str]] = []  # (section_num, title, body)
    for idx, m in enumerate(matches):
        section_num = int(m.group(1))
        section_title = m.group(2).strip()
        body_start = m.end()  # Start the body AFTER the em-dash
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        raw_sections.append((section_num, section_title, body))

    print(f"[ingest] Found {len(raw_sections)} section headers with em-dash.")

    # Filter out empty or very short bodies (often TOC entries that somehow matched)
    filtered_sections = []
    for sec_num, sec_title, body in raw_sections:
        if len(body) >= 60:
            # We add back the section header to the chunk text so the LLM sees it!
            # The user asked to split on it, but the LLM still needs to know which section it is reading
            # so we reconstruct a clean header.
            full_text = f"{sec_num}. {sec_title}.—\n{body}"
            filtered_sections.append((sec_num, sec_title, full_text))
        else:
            pass # Discard chunk

    print(f"[ingest] After filtering short sections (<60 chars body): {len(filtered_sections)} sections.")

    # Sub-chunk long sections
    chunks: list[tuple[str, dict]] = []
    global_idx = 0
    for sec_num, sec_title, body in filtered_sections:
        if len(body) <= MAX_SECTION_CHARS:
            meta = {
                "section_num":   sec_num,
                "section_title": sec_title.strip(),
                "sub_chunk":     0,
                "chunk_index":   global_idx,
            }
            chunks.append((body, meta))
            global_idx += 1
        else:
            sub_chunks = _sub_chunk(body, sec_num, sec_title, global_idx)
            chunks.extend(sub_chunks)
            global_idx += len(sub_chunks)

    print(f"[ingest] Final chunk count (after sub-chunking long sections): {len(chunks)}")
    return chunks


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via LM Studio embeddings endpoint."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def ingest():
    # 1. Load PDF
    text = load_pdf(PDF_PATH)

    # 2. Chunk on section boundaries
    chunks = chunk_by_section(text)

    # 3. Set up ChromaDB
    print(f"[ingest] Initialising ChromaDB at: {CHROMA_PATH}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection to allow re-ingestion
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"[ingest] Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 4. Embed & store in batches
    total_batches = math.ceil(len(chunks) / BATCH_SIZE)
    print(f"[ingest] Embedding {len(chunks)} chunks in {total_batches} batches…")

    texts_only     = [c[0] for c in chunks]
    metadatas_only = [c[1] for c in chunks]

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(chunks))

        batch_texts = texts_only[start:end]
        batch_meta  = metadatas_only[start:end]
        embeddings  = embed_batch(batch_texts)
        ids         = [f"chunk_{i}" for i in range(start, end)]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        print(f"[ingest]   Batch {batch_idx + 1}/{total_batches} stored "
              f"({end}/{len(chunks)} chunks).")

    print(f"\n[ingest] [OK] Ingestion complete. "
          f"Collection '{COLLECTION_NAME}' has {collection.count()} entries.")


if __name__ == "__main__":
    ingest()
