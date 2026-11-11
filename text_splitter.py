from dataclasses import dataclass
from .document_loader import PageDocument

@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: int

def split_pages(pages: list[PageDocument], chunk_size: int = 900, overlap: int = 140) -> list[Chunk]:
    chunks = []
    for page in pages:
        text, start, index = page.text.strip(), 0, 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            if end < len(text):
                boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(piece, page.source, page.page, index))
                index += 1
            if end >= len(text): break
            start = max(start + 1, end - overlap)
    return chunks
