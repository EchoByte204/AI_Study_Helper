import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    page_number: int
    chunk_index: int
    content: str


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: list[str], chunk_size: int, overlap: int) -> list[TextChunk]:
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[TextChunk] = []
    index = 0
    for page_number, page in enumerate(pages, start=1):
        text = normalize_text(page)
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                # Prefer a nearby word boundary so chunks remain readable.
                boundary = text.rfind(" ", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            piece = text[start:end].strip()
            if piece:
                chunks.append(TextChunk(page_number, index, piece))
                index += 1
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
