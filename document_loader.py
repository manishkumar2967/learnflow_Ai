"""PDF extraction with page-level provenance."""
from dataclasses import dataclass
from io import BytesIO
from pypdf import PdfReader

@dataclass
class PageDocument:
    text: str
    source: str
    page: int

def load_pdf(uploaded_file) -> list[PageDocument]:
    try:
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    except Exception as exc:
        raise ValueError(f"Could not read '{uploaded_file.name}': {exc}") from exc
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(PageDocument(text=text, source=uploaded_file.name, page=number))
    if not pages:
        raise ValueError(f"'{uploaded_file.name}' contains no extractable text (it may be a scanned PDF).")
    return pages
