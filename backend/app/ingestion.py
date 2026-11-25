import hashlib
import logging
import posixpath
import re
import time
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.embeddings import EmbeddingService
from app.models import Chunk, Document, Figure
from app.text_processing import chunk_pages

logger = logging.getLogger("rag_tutor.ingestion")
EXTRACTION_VERSION = 2


class IngestionError(ValueError):
    pass


@dataclass
class Visual:
    page_number: int
    image_bytes: bytes
    media_type: str
    description: str


ProgressCallback = Callable[[int, str], None]


def _extract_pdf_visuals(
    content: bytes,
    pages: list[str],
    filename: str,
    progress_callback: ProgressCallback | None = None,
    pdf_reader=None,
) -> list[Visual]:
    from pypdf import PdfReader

    try:
        import pymupdf

        rendered_pdf = pymupdf.open(stream=content, filetype="pdf")
    except Exception:
        logger.warning("PyMuPDF could not render %s; extracting only embedded images", filename, exc_info=True)
        pymupdf = None
        rendered_pdf = None
    visuals: list[Visual] = []
    total_bytes = 0
    seen_pdf_images: set[bytes] = set()
    seen_pdf_xrefs: set[int] = set()
    reader = pdf_reader or PdfReader(BytesIO(content), strict=False)
    for page_index, page in enumerate(reader.pages):
        if progress_callback:
            progress_callback(26 + round(page_index / max(len(reader.pages), 1) * 14), f"Extracting figures from PDF page {page_index + 1} of {len(reader.pages)}")
        if len(visuals) >= 100:
            break
        page_text = pages[page_index][:1200].strip()
        page_has_embedded_image = False
        try:
            # MuPDF can list image references without decoding the image bytes.
            # Prefer it when available so repeated logos aren't decompressed on
            # every page; use pypdf as a fallback when MuPDF is unavailable.
            page_images = list(page.images) if rendered_pdf is None else []
        except Exception:
            logger.warning("Could not enumerate embedded images on PDF page %s", page_index + 1, exc_info=True)
            page_images = []
        for image in page_images:
            if len([v for v in visuals if v.page_number == page_index + 1]) >= 12:
                continue
            try:
                image_data = image.data
            except Exception:
                logger.warning("Could not decode an embedded image on PDF page %s", page_index + 1, exc_info=True)
                continue
            if len(image_data) > 8 * 1024 * 1024:
                continue
            media_type = _image_media_type(image.name)
            if media_type and media_type != "image/svg+xml":
                signature = hashlib.sha256(image_data).digest()
                page_has_embedded_image = True
                if signature not in seen_pdf_images and total_bytes + len(image_data) <= 50 * 1024 * 1024:
                    seen_pdf_images.add(signature)
                    visuals.append(Visual(page_index + 1, image_data, media_type, f"Figure from {filename}, page {page_index + 1}. Nearby page text: {page_text}"))
                    total_bytes += len(image_data)
        if rendered_pdf is not None and len(visuals) < 100:
            try:
                visual_page = rendered_pdf[page_index]
                # Some PDFs wrap image objects in forms that pypdf doesn't expose.
                # Use MuPDF's image catalog as the primary extraction path.
                if not page_has_embedded_image:
                    for image_info in visual_page.get_images(full=True):
                        if sum(1 for item in visuals if item.page_number == page_index + 1) >= 12:
                            break
                        xref = image_info[0]
                        if xref in seen_pdf_xrefs:
                            page_has_embedded_image = True
                            continue
                        seen_pdf_xrefs.add(xref)
                        extracted = rendered_pdf.extract_image(xref)
                        image_data = extracted["image"]
                        media_type = _image_media_type(f"image.{extracted.get('ext', '')}")
                        signature = hashlib.sha256(image_data).digest()
                        if media_type and media_type != "image/svg+xml" and len(image_data) <= 8 * 1024 * 1024:
                            page_has_embedded_image = True
                            if signature not in seen_pdf_images and total_bytes + len(image_data) <= 50 * 1024 * 1024:
                                seen_pdf_images.add(signature)
                                visuals.append(Visual(page_index + 1, image_data, media_type, f"Figure from {filename}, page {page_index + 1}. Nearby page text: {page_text}"))
                                total_bytes += len(image_data)
                page_visual_count = sum(1 for item in visuals if item.page_number == page_index + 1)
                is_exported_slide_deck = filename.lower().endswith(".pptx.pdf")
                # Rendering every exported slide is expensive. Extract embedded
                # images first, and render only pages that have no usable image.
                should_render = False
                if page_visual_count == 0 and not page_has_embedded_image:
                    drawing_count = len(visual_page.get_drawings())
                    # A page full of text needs no screenshot. Render likely
                    # diagram pages only; this matters for exported slide PDFs.
                    drawing_threshold = 8 if is_exported_slide_deck else 3
                    should_render = drawing_count >= drawing_threshold
                if should_render:
                    scale = 1.0 if is_exported_slide_deck else 1.05
                    rendered = visual_page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")
                    if len(rendered) <= 8 * 1024 * 1024 and total_bytes + len(rendered) <= 50 * 1024 * 1024:
                        label = "Rendered slide" if is_exported_slide_deck else "Rendered page figure"
                        visuals.append(Visual(page_index + 1, rendered, "image/png", f"{label} from {filename}, page {page_index + 1}. Page text: {page_text}"))
                        total_bytes += len(rendered)
            except Exception:
                logger.warning("Could not render vector figures on PDF page %s", page_index + 1, exc_info=True)
    if rendered_pdf is not None:
        rendered_pdf.close()
    return visuals


def _extract_pptx(
    content: bytes, filename: str, progress_callback: ProgressCallback | None = None
) -> tuple[list[str], list[Visual]]:
    try:
        archive = zipfile.ZipFile(BytesIO(content))
    except Exception as exc:
        raise IngestionError("The file could not be read as a PowerPoint presentation") from exc
    pages: list[str] = []
    visuals: list[Visual] = []
    total_bytes = 0
    seen_images: set[str] = set()
    seen_targets: set[str] = set()
    namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    chart_namespace = {"a": namespace["a"], "r": namespace["r"], "c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
    slide_names = sorted((name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)), key=lambda name: int(re.search(r"slide(\d+)", name).group(1)))
    for slide_number, slide_name in enumerate(slide_names, start=1):
        if progress_callback:
            progress_callback(12 + round((slide_number - 1) / max(len(slide_names), 1) * 28), f"Reading slide {slide_number} of {len(slide_names)}")
        root = ElementTree.fromstring(archive.read(slide_name))
        slide_text = "\n".join((item.text or "") for item in root.findall(".//a:t", namespace)).strip()
        rels_name = posixpath.join(posixpath.dirname(slide_name), "_rels", posixpath.basename(slide_name) + ".rels")
        relationships = {}
        if rels_name in archive.namelist():
            rel_root = ElementTree.fromstring(archive.read(rels_name))
            relationships = {
                rel.attrib["Id"]: (
                    rel.attrib["Target"].lstrip("/")
                    if rel.attrib["Target"].startswith("/")
                    else posixpath.normpath(posixpath.join(posixpath.dirname(slide_name), rel.attrib["Target"]))
                )
                for rel in rel_root
            }
        structured_content: list[str] = []
        for table_number, table in enumerate(root.findall(".//a:tbl", namespace), start=1):
            table_rows = []
            for row in table.findall("a:tr", namespace):
                cells = [" ".join(text.text or "" for text in cell.findall(".//a:t", namespace)).strip() for cell in row.findall("a:tc", namespace)]
                table_rows.append(" | ".join(cells))
            if table_rows:
                structured_content.append(f"Table {table_number}:\n" + "\n".join(table_rows))
        chart_refs = root.findall(".//c:chart", chart_namespace)
        for chart_number, chart_ref in enumerate(chart_refs, start=1):
            relationship_id = chart_ref.attrib.get(f"{{{namespace['r']}}}id", "")
            chart_target = relationships.get(relationship_id)
            if not chart_target or chart_target not in archive.namelist():
                continue
            try:
                chart_root = ElementTree.fromstring(archive.read(chart_target))
                chart_title = " ".join((item.text or "").strip() for item in chart_root.findall(".//c:title//a:t", chart_namespace)).strip()
                chart_details = [f"Chart {chart_number}" + (f": {chart_title}" if chart_title else "")]
                for series_number, series in enumerate(chart_root.findall(".//c:ser", chart_namespace), start=1):
                    labels = [value.text or "" for value in series.findall(".//c:cat//c:v", chart_namespace)]
                    values = [value.text or "" for value in series.findall(".//c:val//c:v", chart_namespace)]
                    names = [value.text or "" for value in series.findall(".//c:tx//c:v", chart_namespace)]
                    series_name = names[0] if names else f"Series {series_number}"
                    chart_details.append(f"{series_name}: " + "; ".join(f"{label} = {value}" for label, value in zip(labels, values)) if labels and len(labels) == len(values) else f"{series_name}: " + ", ".join(values))
                structured_content.append("\n".join(chart_details))
            except ElementTree.ParseError:
                logger.warning("Could not parse chart %s on slide %s", chart_target, slide_number)
        if structured_content:
            slide_text = "\n\n".join(part for part in [slide_text, *structured_content] if part)
        pages.append(slide_text)
        for picture in root.findall(".//a:blip", namespace):
            if len(visuals) >= 100:
                break
            target = relationships.get(picture.attrib.get(f"{{{namespace['r']}}}embed", ""))
            if not target or target in seen_targets or target not in archive.namelist():
                continue
            seen_targets.add(target)
            info = archive.getinfo(target)
            if info.file_size > 8 * 1024 * 1024 or total_bytes + info.file_size > 50 * 1024 * 1024:
                continue
            media_type = _image_media_type(target)
            image_data = archive.read(target)
            image_hash = hashlib.sha256(image_data).hexdigest()
            if media_type and image_hash not in seen_images and info.file_size >= 2048:
                seen_images.add(image_hash)
                visuals.append(Visual(slide_number, image_data, media_type, f"Figure from {filename}, slide {slide_number}. Nearby slide text: {slide_text[:1200]}"))
                total_bytes += len(image_data)
    return pages, visuals


def _image_media_type(filename: str) -> str | None:
    extension = Path(filename).suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif", ".svg": "image/svg+xml"}.get(extension)


def ingest_document(
    db: Session,
    filename: str,
    content: bytes,
    embeddings: EmbeddingService | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Document:
    ingestion_started = time.perf_counter()
    settings = get_settings()
    if not content:
        raise IngestionError("The uploaded file is empty")
    extension = Path(filename).suffix.lower()
    if extension not in {".pdf", ".pptx"}:
        raise IngestionError("Upload a PDF or PowerPoint (.pptx) file")
    sha256 = hashlib.sha256(content).hexdigest()
    if progress_callback:
        progress_callback(12, "Checking the document index")
    database_check_started = time.perf_counter()
    existing = db.scalar(select(Document).where(Document.sha256 == sha256))
    logger.info("Checked duplicate index for %s in %.2fs", filename, time.perf_counter() - database_check_started)
    if existing and existing.ingestion_version >= EXTRACTION_VERSION:
        if progress_callback:
            progress_callback(100, "This file is already indexed")
        return existing
    old_media_paths = list(db.scalars(select(Figure.storage_path).where(Figure.document_id == existing.id))) if existing else []
    extraction_started = time.perf_counter()
    try:
        if extension == ".pdf":
            from pypdf import PdfReader

            if progress_callback:
                progress_callback(12, "Opening PDF and reading its text layer")
            reader = PdfReader(BytesIO(content), strict=False)
            pages = []
            for page_index, page in enumerate(reader.pages, start=1):
                pages.append(page.extract_text() or "")
                if progress_callback:
                    progress_callback(12 + round(page_index / max(len(reader.pages), 1) * 12), f"Reading PDF page {page_index} of {len(reader.pages)}")
            try:
                visuals = _extract_pdf_visuals(content, pages, filename, progress_callback, reader)
            except Exception:
                logger.warning("Figure extraction failed for %s; continuing with page text", filename, exc_info=True)
                visuals = []
        else:
            pages, visuals = _extract_pptx(content, filename, progress_callback)
    except IngestionError:
        raise
    except ModuleNotFoundError as exc:
        logger.exception("A required parser dependency is missing while reading %s", filename)
        raise IngestionError(
            f"The API environment is missing the required parser package '{exc.name or 'unknown'}'. Install the project dependencies in the API's Python environment."
        ) from exc
    except Exception as exc:
        kind = "PDF" if extension == ".pdf" else "PowerPoint"
        logger.exception("Could not read uploaded %s file %s", kind, filename)
        raise IngestionError(
            f"Could not read this {kind} file ({type(exc).__name__}). It may be damaged, password-protected, or saved in an unsupported format."
        ) from exc
    if (not pages or not any(page.strip() for page in pages)) and not visuals:
        raise IngestionError("No selectable text or extractable figures were found. Scanned PDFs need OCR before upload")
    logger.info(
        "Extracted %s: %s pages, %s figures in %.2fs",
        filename,
        len(pages),
        len(visuals),
        time.perf_counter() - extraction_started,
    )
    pieces = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if progress_callback:
        progress_callback(42, f"Prepared {len(pages)} pages and {len(pieces)} text passages")
    embedding_service = embeddings or EmbeddingService()
    batch_size = max(1, settings.embedding_batch_size)
    embedding_texts = [piece.content for piece in pieces] + [visual.description for visual in visuals]
    batches = [embedding_texts[start : start + batch_size] for start in range(0, len(embedding_texts), batch_size)]
    vectors: list[list[float]] = []
    embedding_started = time.perf_counter()
    if progress_callback:
        progress_callback(48, f"Creating embeddings for {len(embedding_texts)} text and image items")
    if settings.openai_api_key and len(batches) > 1:
        ordered_vectors: list[list[list[float]] | None] = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=min(4, max(1, settings.embedding_workers))) as executor:
            futures = {executor.submit(embedding_service.embed, batch): index for index, batch in enumerate(batches)}
            for completed_count, future in enumerate(as_completed(futures), start=1):
                batch_index = futures[future]
                ordered_vectors[batch_index] = future.result()
                if progress_callback:
                    progress_callback(48 + round(completed_count / len(batches) * 42), f"Embedded batch {completed_count} of {len(batches)}")
        vectors = [vector for batch_vectors in ordered_vectors if batch_vectors for vector in batch_vectors]
    else:
        for batch_index, batch in enumerate(batches, start=1):
            vectors.extend(embedding_service.embed(batch))
            if progress_callback:
                progress_callback(48 + round(batch_index / max(len(batches), 1) * 42), f"Embedding batch {batch_index} of {len(batches)}")

    logger.info(
        "Embedded %s: %s items in %.2fs",
        filename,
        len(embedding_texts),
        time.perf_counter() - embedding_started,
    )

    document = existing or Document(filename=filename[:512], sha256=sha256)
    if existing:
        document.filename = filename[:512]
        document.page_count = len(pages)
        document.chunk_count = len(pieces)
        document.figure_count = len(visuals)
        document.visual_indexed = bool(visuals)
        document.ingestion_version = EXTRACTION_VERSION
        db.execute(delete(Chunk).where(Chunk.document_id == existing.id))
        db.execute(delete(Figure).where(Figure.document_id == existing.id))
    else:
        document.page_count = len(pages)
        document.chunk_count = len(pieces)
        document.figure_count = len(visuals)
        document.visual_indexed = bool(visuals)
        document.ingestion_version = EXTRACTION_VERSION
        db.add(document)
        db.flush()
    media_root = Path(settings.media_dir).resolve() / document.id
    media_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        if progress_callback:
            progress_callback(93, "Saving document and figure index")
        db.add_all([
            Chunk(document_id=document.id, page_number=piece.page_number, chunk_index=piece.chunk_index,
                  content=piece.content, embedding=vector)
            for piece, vector in zip(pieces, vectors[: len(pieces)], strict=True)
        ])
        for index, visual in enumerate(visuals):
            extension = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif", "image/svg+xml": ".svg"}[visual.media_type]
            path = media_root / f"figure-v{EXTRACTION_VERSION}-{index + 1}{extension}"
            path.write_bytes(visual.image_bytes)
            written.append(path)
            db.add(Figure(document_id=document.id, page_number=visual.page_number, figure_index=index + 1,
                          media_type=visual.media_type, storage_path=str(path), description=visual.description,
                          embedding=vectors[len(pieces) + index]))
        db.commit()
    except Exception:
        db.rollback()
        for path in written:
            path.unlink(missing_ok=True)
        media_root.rmdir() if media_root.exists() and not any(media_root.iterdir()) else None
        raise
    for old_path in old_media_paths:
        try:
            Path(old_path).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove replaced figure file: %s", old_path)
    db.refresh(document)
    logger.info("Indexed %s in %.2fs", filename, time.perf_counter() - ingestion_started)
    if progress_callback:
        progress_callback(100, "Document ready")
    return document


def ingest_pdf(db: Session, filename: str, content: bytes, embeddings: EmbeddingService | None = None) -> Document:
    """Backward-compatible alias for callers that still use the PDF-specific name."""
    if not filename.lower().endswith(".pdf"):
        raise IngestionError("Upload a PDF file")
    return ingest_document(db, filename, content, embeddings)
