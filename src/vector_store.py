"""FAISS-backed vector store for resume chunks."""

import json
import os

import faiss
import numpy as np

from src.embeddings import EMBEDDING_DIM


class ResumeVectorStore:
    """Stores chunk embeddings in a FAISS index alongside chunk metadata/text."""

    def __init__(self, dim=EMBEDDING_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
        self.metadata = []  # parallel list: metadata[i] corresponds to vector i

    def add_chunks(self, chunks):
        """Add chunks (each with an "embedding" key) to the index.

        Any other keys on a chunk (chunk_id, section, page, source, text, ...) are kept
        as metadata, so this works for both resume chunks and knowledge base chunks.
        """
        vectors = np.array([chunk["embedding"] for chunk in chunks], dtype="float32")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        for chunk in chunks:
            self.metadata.append({k: v for k, v in chunk.items() if k != "embedding"})

    def search(self, query_embedding, top_k=5):
        """Return the top_k most similar chunks for a query embedding."""
        if self.index.ntotal == 0:
            return []
        vector = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(score)
            results.append(result)
        return results

    def save(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))
        with open(os.path.join(dir_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, dir_path):
        store = cls.__new__(cls)
        store.index = faiss.read_index(os.path.join(dir_path, "index.faiss"))
        store.dim = store.index.d
        with open(os.path.join(dir_path, "metadata.json"), "r", encoding="utf-8") as f:
            store.metadata = json.load(f)
        return store
