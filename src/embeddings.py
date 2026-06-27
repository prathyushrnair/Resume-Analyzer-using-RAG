"""Embedding generation for resume chunks using the Gemini Embedding API."""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # output_dimensionality requested from the API


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use the Gemini Embedding API.")
    return genai.Client(api_key=api_key)


def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", client=None):
    """Embed a list of strings, returning a list of float vectors (one per text)."""
    if not texts:
        return []
    client = client or get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return [embedding.values for embedding in response.embeddings]


def embed_chunks(chunks, client=None):
    """Embed resume chunks (as produced by pdf_processing.chunk_resume).

    Returns the same chunks with an added "embedding" key.
    """
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", client=client)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def embed_query(query, client=None):
    """Embed a single user query string for similarity search."""
    return embed_texts([query], task_type="RETRIEVAL_QUERY", client=client)[0]
