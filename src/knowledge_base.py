"""Loads and chunks the resume knowledge base (ATS rules, writing guidelines, rubrics)."""

import glob
import os
import re

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")


def load_knowledge_base_chunks(kb_dir=KNOWLEDGE_BASE_DIR):
    """Split each markdown file in the knowledge base into paragraph-level chunks.

    Returns a list of dicts: {chunk_id, source, text}.
    """
    chunks = []
    chunk_id = 1
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.md"))):
        source = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        for paragraph in paragraphs:
            if paragraph.startswith("#"):
                continue  # skip standalone headings
            chunks.append({"chunk_id": chunk_id, "source": source, "text": paragraph})
            chunk_id += 1
    return chunks


def add_job_description(text, chunks=None):
    """Wrap an optional target job description as a knowledge base chunk."""
    chunks = chunks if chunks is not None else []
    next_id = max((c["chunk_id"] for c in chunks), default=0) + 1
    chunks.append({"chunk_id": next_id, "source": "job_description", "text": text.strip()})
    return chunks
