"""PDF parsing and semantic chunking for resumes.

Extracts text from a resume PDF, cleans it, and splits it into
semantic sections (Summary, Education, Experience, Projects, etc.)
with metadata (section name, page number, chunk id).
"""

import re

import fitz  # PyMuPDF

# Canonical section name -> header aliases that may appear in a resume.
SECTION_ALIASES = {
    "Summary": ["summary", "professional summary", "objective", "profile"],
    "Education": ["education", "academic background"],
    "Experience": ["experience", "work experience", "professional experience", "employment history"],
    "Projects": ["projects", "personal projects", "academic projects"],
    "Skills": ["skills", "technical skills", "skills & tools", "core competencies"],
    "Certifications": ["certifications", "certificates", "licenses"],
    "Achievements": ["achievements", "awards", "honors", "accomplishments"],
}

# Flatten alias -> canonical name for fast lookup.
ALIAS_TO_SECTION = {
    alias: section for section, aliases in SECTION_ALIASES.items() for alias in aliases
}

MAX_HEADER_LENGTH = 40


def extract_lines_with_pages(pdf_path):
    """Return a list of (page_number, line_text) tuples for the whole PDF."""
    lines = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text()
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if line:
                    lines.append((page_index, line))
    return lines


def detect_section(line):
    """Return the canonical section name if line looks like a section header, else None."""
    if len(line) > MAX_HEADER_LENGTH:
        return None
    normalized = re.sub(r"[^a-zA-Z& ]", "", line).strip().lower()
    return ALIAS_TO_SECTION.get(normalized)


def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def chunk_resume(pdf_path):
    """Parse a resume PDF into a list of semantic chunks.

    Each chunk is a dict: {chunk_id, section, page, text}.
    Content before the first recognized header is kept under "Header".
    """
    lines = extract_lines_with_pages(pdf_path)

    chunks = []
    current_section = "Header"
    current_page = lines[0][0] if lines else 1
    current_text_lines = []
    chunk_id = 1

    def flush():
        nonlocal chunk_id
        text = clean_text("\n".join(current_text_lines))
        if text:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "section": current_section,
                    "page": current_page,
                    "text": text,
                }
            )
            chunk_id += 1

    for page_num, line in lines:
        section = detect_section(line)
        if section:
            flush()
            current_section = section
            current_page = page_num
            current_text_lines = []
        else:
            current_text_lines.append(line)

    flush()
    return chunks
