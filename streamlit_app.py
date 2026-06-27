"""Streamlit chatbot UI: upload a resume PDF, then chat with an AI HR recruiter about it."""

import os
import tempfile

import streamlit as st

from src.embeddings import get_client
from src.knowledge_base import add_job_description, load_knowledge_base_chunks
from src.pdf_processing import chunk_resume
from src.rag_pipeline import answer_query, build_resume_store
from src.embeddings import embed_chunks
from src.vector_store import ResumeVectorStore

st.set_page_config(page_title="Resume Analyzer", page_icon="📄")
st.title("📄 Resume Analyzer")
st.caption("Upload your resume and chat with an AI recruiter for ATS-focused feedback.")


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    return get_client()


@st.cache_resource(show_spinner=False)
def get_knowledge_base_store(_client):
    kb_chunks = load_knowledge_base_chunks()
    kb_chunks = embed_chunks(kb_chunks, client=_client)
    store = ResumeVectorStore()
    store.add_chunks(kb_chunks)
    return store


if "messages" not in st.session_state:
    st.session_state.messages = []
if "resume_store" not in st.session_state:
    st.session_state.resume_store = None
if "jd_store" not in st.session_state:
    st.session_state.jd_store = None

try:
    client = get_gemini_client()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

kb_store = get_knowledge_base_store(client)

with st.sidebar:
    st.header("Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    if uploaded_file is not None:
        if st.session_state.get("uploaded_file_name") != uploaded_file.name:
            with st.spinner("Parsing and indexing your resume..."):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    resume_chunks = chunk_resume(tmp_path)
                finally:
                    os.remove(tmp_path)

                if not resume_chunks:
                    st.error("Could not extract any text from this PDF.")
                else:
                    st.session_state.resume_store = build_resume_store(resume_chunks, client=client)
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.messages = []
                    st.success(f"Indexed {len(resume_chunks)} resume sections.")

    st.divider()
    st.header("Target Job Description (optional)")
    job_description = st.text_area("Paste a job description to compare against", height=150)
    if st.button("Use job description", disabled=not job_description.strip()):
        with st.spinner("Indexing job description..."):
            jd_chunks = add_job_description(job_description)
            jd_chunks = embed_chunks(jd_chunks, client=client)
            jd_store = ResumeVectorStore()
            jd_store.add_chunks(jd_chunks)
            st.session_state.jd_store = jd_store
        st.success("Job description added to context.")

if st.session_state.resume_store is None:
    st.info("Upload a resume PDF in the sidebar to get started.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about your resume, e.g. 'Is this ATS friendly?'")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    kb_stores = [kb_store] + ([st.session_state.jd_store] if st.session_state.jd_store else [])

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = answer_query(
                prompt,
                st.session_state.resume_store,
                kb_stores,
                client=client,
            )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
