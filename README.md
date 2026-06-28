# 📄 Resume Analyzer using RAG and Gemini

An AI-powered Resume Analyzer that uses **Retrieval-Augmented Generation (RAG)** and the
**Google Gemini API** to give personalized, ATS-focused feedback on resumes through a
conversational chatbot.

Instead of just reading the uploaded resume, the system retrieves the relevant resume
sections together with resume-writing best practices, ATS guidelines, and (optionally) a
target job description, then grounds Gemini's answer in that retrieved context.



## Features
1.Upload a resume in PDF format
2. Semantic chunking by resume section (Summary, Education, Experience, Projects, Skills, ...)
3. Gemini embeddings for resume chunks and knowledge base content
4. Semantic retrieval with a local FAISS vector index
5. ATS-focused evaluation grounded in a small knowledge base (ATS rules, writing guidelines, HR rubrics)
6. Optional job description matching
7. Conversational chatbot interface (Gradio for Hugging Face Spaces, Streamlit for local use)



## Architecture
```text
Resume PDF
    │
    ▼
PDF Processing (src/pdf_processing.py)
    │  extract text per page, detect section headers, clean
    ▼
Semantic Chunks  { chunk_id, section, page, text }
    │
    ▼
Gemini Embeddings (src/embeddings.py)
    │
    ▼
FAISS Vector Store (src/vector_store.py)  ◄── also stores the
    │                                          knowledge base +
    │                                          optional job description
    ▼
User Question
    │
    ▼
Query Embedding → Similarity Search (resume store + knowledge base store)
    │
    ▼
Prompt Construction (src/rag_pipeline.py)
    │
    ▼
Gemini API (generation)
    │
    ▼
Chatbot Response (app.py / streamlit_app.py)
```



## Project Structure

```text
.
├── app.py                      # Gradio chatbot UI (entry point for Hugging Face Spaces)
├── streamlit_app.py            # Streamlit chatbot UI (for local use)
├── requirements.txt
├── .env.example                 # Template for required environment variables
├── knowledge_base/             # ATS rules, writing guidelines, HR rubrics (markdown)
│   ├── ats_best_practices.md
│   ├── resume_writing_guidelines.md
│   └── hr_evaluation_rubrics.md
└── src/
    ├── pdf_processing.py       # PDF text extraction + semantic chunking
    ├── embeddings.py           # Gemini embedding API wrapper
    ├── vector_store.py         # FAISS-backed vector store (ResumeVectorStore)
    ├── knowledge_base.py       # Loads/chunks knowledge_base/*.md, optional job description
    └── rag_pipeline.py         # Retrieval + prompt construction + generation
```



## Setup

### 1. Clone and create a virtual environment

```bash
git clone <this-repo>
cd "Resume Analyzer using RAG"
python -m venv venv
```

Activate it:

```powershell
# PowerShell
.\venv\Scripts\Activate.ps1
```

```bash
# Git Bash / macOS / Linux
source venv/Scripts/activate   # Windows
source venv/bin/activate       # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your Gemini API key

Copy the example env file and fill in your key:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get a key from [Google AI Studio](https://aistudio.google.com/). `.env` is gitignored and
loaded automatically via `python-dotenv`.



## Running the app

### Streamlit app running and deployment 


Opens at `http://127.0.0.1:7860`.

### Streamlit (local alternative UI)

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

### Using either UI

1. Upload a resume PDF in the sidebar/left panel — it's parsed, chunked, and embedded.
2. (Optional) Paste a target job description and click "Use job description" to ground
   answers in that context too.
3. Chat: ask things like
   - "Review my resume."
   - "Is my resume ATS friendly?"
   - "How can I improve my projects?"
   - "How well does my resume match this job description?"



## Deploying to Streamlit  

1. Create a new Space with the streamlit web interface.
2. Push this repo's contents to the Space (entry point is `app.py`, declared in this
   README's YAML frontmatter).
3. In the Space's **Settings → Variables and secrets**, add a secret named
   `GEMINI_API_KEY` with your Gemini API key — there's no `.env` file in a Space, so the
   secret becomes an environment variable automatically, which `src/embeddings.py` reads.



## Knowledge Base

The `knowledge_base/` directory holds the reference material retrieved alongside resume
chunks: ATS best practices, resume writing guidelines, and HR evaluation rubrics. Add or
edit markdown files there to expand what the chatbot can ground its answers in —
`src/knowledge_base.py` chunks each file by paragraph automatically.



## Technology Stack

| Component         | Technology                  |
|--|-|
| Chat UI            |  Streamlit                 |
| LLM                | Google Gemini (`gemini-2.5-flash`) |
| Embeddings         | Gemini Embedding API (`gemini-embedding-001`) |
| PDF Parsing        | PyMuPDF                    |
| Vector Database    | FAISS                      |
| Language           | Python                     |



## Future Improvements

- Multi-resume comparison
- Cover letter generation
- Resume version tracking
- Resume ranking against other applicants
- AI-generated interview questions
- One-click resume rewriting



## Why RAG?

A resume is short enough that an LLM could often analyze it directly. RAG adds value by
retrieving external knowledge — ATS rules, resume writing guidelines, recruiter evaluation
criteria, and a target job description — and combining it with the resume content. This
makes the chatbot's recommendations more explainable, consistent, and grounded in
established best practices rather than relying solely on the model's internal knowledge.
Deployment Link :
https://prathyushrnair-resume-analyzer-using-rag-app-ej9qpt.streamlit.app/
