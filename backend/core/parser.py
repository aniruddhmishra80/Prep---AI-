"""
Turn an uploaded PDF into plain text.

Interview line: "PyPDFLoader reads the text layer of the PDF. It gives back one
LangChain Document per page, which I join into a single string. Known limit: a
scanned image-only resume returns nothing, because there is no text layer to
read. Handling that needs OCR, which I kept out of scope."
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf_text(path: str | Path) -> str:
    """Extract all text from a PDF file on disk."""
    documents = PyPDFLoader(str(path)).load()
    text = "\n".join(doc.page_content for doc in documents)
    return normalise(text)


def normalise(text: str) -> str:
    """Collapse runaway whitespace that PDF extraction leaves behind."""
    return " ".join((text or "").split()).strip()


def save_upload(file_bytes: bytes, filename: str, upload_dir: Path) -> Path:
    """Write an uploaded file to disk and return its path."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / filename
    destination.write_bytes(file_bytes)
    return destination
