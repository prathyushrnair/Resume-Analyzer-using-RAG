"""RAG pipeline: retrieval + prompt construction + generation for resume analysis."""

import os

from dotenv import load_dotenv
from google import genai

from src.embeddings import embed_chunks, embed_query, get_client
from src.knowledge_base import load_knowledge_base_chunks
from src.vector_store import ResumeVectorStore

load_dotenv()

GENERATION_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are an experienced HR recruiter and resume coach.

Evaluate the candidate's resume using the provided resume excerpts and reference material.
Ground every claim in the retrieved context below; do not invent details that aren't
present in the resume.

Provide your response with these sections:
- Strengths
- Weaknesses
- ATS Compatibility
- Missing Keywords
- Suggestions

Be specific and concise. Reference the resume's actual content where relevant."""


def build_knowledge_base_store(client=None):
    """Embed and index all knowledge base documents into a FAISS store."""
    client = client or get_client()
    kb_chunks = load_knowledge_base_chunks()
    kb_chunks = embed_chunks(kb_chunks, client=client)
    store = ResumeVectorStore()
    store.add_chunks(kb_chunks)
    return store


def build_resume_store(resume_chunks, client=None):
    """Embed and index resume chunks (from pdf_processing.chunk_resume) into a FAISS store."""
    client = client or get_client()
    resume_chunks = embed_chunks(resume_chunks, client=client)
    store = ResumeVectorStore()
    store.add_chunks(resume_chunks)
    return store


def retrieve_context(query, resume_store, kb_store, top_k_resume=5, top_k_kb=3, client=None):
    """Embed the query and retrieve the most relevant resume and knowledge base chunks.

    kb_store may be a single ResumeVectorStore or a list of them (e.g. the shared
    knowledge base plus a per-session job description store); results are merged by score.
    """
    client = client or get_client()
    query_embedding = embed_query(query, client=client)
    resume_results = resume_store.search(query_embedding, top_k=top_k_resume)

    kb_stores = kb_store if isinstance(kb_store, (list, tuple)) else [kb_store]
    kb_results = [r for store in kb_stores for r in store.search(query_embedding, top_k=top_k_kb)]
    kb_results.sort(key=lambda r: r["score"], reverse=True)
    kb_results = kb_results[:top_k_kb]

    return resume_results, kb_results


def build_prompt(query, resume_results, kb_results):
    """Combine retrieved resume chunks and knowledge base chunks into a single prompt."""
    resume_context = "\n\n".join(
        f"[{r['section']} - page {r['page']}]\n{r['text']}" for r in resume_results
    ) or "(no resume content retrieved)"

    reference_context = "\n\n".join(
        f"[{r['source']}]\n{r['text']}" for r in kb_results
    ) or "(no reference material retrieved)"

    return f"""{SYSTEM_PROMPT}

Resume Context:
{resume_context}

Reference Material:
{reference_context}

User Question:
{query}"""


def answer_query(query, resume_store, kb_store, top_k_resume=5, top_k_kb=3, client=None):
    """Full RAG round trip: retrieve context, build prompt, call Gemini, return the response text."""
    client = client or get_client()
    resume_results, kb_results = retrieve_context(
        query, resume_store, kb_store, top_k_resume=top_k_resume, top_k_kb=top_k_kb, client=client
    )
    prompt = build_prompt(query, resume_results, kb_results)
    response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
    return response.text
