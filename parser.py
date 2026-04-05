"""
PDF ingestion module — extracts three modalities from a PDF:

  1. Text   → split into overlapping chunks
  2. Tables → converted to Markdown for readability
  3. Images → extracted as PNG files (captioned separately)

Libraries used:
  - PyMuPDF (fitz) for text and image extraction
  - pdfplumber   for high-fidelity table extraction
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from src.config import settings
from src.models import ChunkType, DocumentChunk

logger = logging.getLogger(__name__)

# Minimum image area (px²) to skip tiny decorative icons
MIN_IMAGE_AREA = 4000


class PDFParser:
    """Parses a PDF and returns a list of DocumentChunk objects."""

    def __init__(self, image_cache_dir: Path | None = None) -> None:
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        # Temporary folder for extracted images
        self.image_cache_dir = image_cache_dir or Path(".cache/images")
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────────────

    def parse(self, pdf_path: Path) -> list[DocumentChunk]:
        """Extract all chunks from *pdf_path*.

        Returns a flat list of DocumentChunk with types TEXT, TABLE, IMAGE.
        """
        filename = pdf_path.name
        chunks: list[DocumentChunk] = []

        logger.info("Parsing %s", filename)

        # Tables first (pdfplumber is more accurate than fitz for tables)
        table_page_indices = set()
        table_chunks = self._extract_tables(pdf_path, filename)
        chunks.extend(table_chunks)
        # Record which pages had tables so we can avoid re-embedding that text
        for tc in table_chunks:
            table_page_indices.add(tc.page_number)

        # Text extraction (fitz is fast and accurate for prose)
        text_chunks = self._extract_text(
            pdf_path, filename, skip_pages=table_page_indices
        )
        chunks.extend(text_chunks)

        # Image extraction (fitz)
        image_chunks_raw = self._extract_images(pdf_path, filename)
        chunks.extend(image_chunks_raw)

        logger.info(
            "Parsed %s → %d text | %d table | %d image chunks",
            filename,
            len(text_chunks),
            len(table_chunks),
            len(image_chunks_raw),
        )
        return chunks

    # ── Text extraction ───────────────────────────────────────────────────────

    def _extract_text(
        self, pdf_path: Path, filename: str, skip_pages: set[int]
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        doc = fitz.open(str(pdf_path))

        for page_num, page in enumerate(doc, start=1):
            if page_num in skip_pages:
                continue  # Page dominated by table — skip to avoid duplication

            text = page.get_text("text")
            text = self._clean_text(text)
            if not text.strip():
                continue

            for chunk_text in self._split_text(text):
                if len(chunk_text.strip()) < 30:
                    continue  # Skip noise fragments
                chunks.append(
                    DocumentChunk(
                        chunk_id=self._make_id(filename, page_num, chunk_text),
                        source_file=filename,
                        page_number=page_num,
                        chunk_type=ChunkType.TEXT,
                        content=chunk_text.strip(),
                    )
                )
        doc.close()
        return chunks

    # ── Table extraction ──────────────────────────────────────────────────────

    def _extract_tables(self, pdf_path: Path, filename: str) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    md = self._table_to_markdown(table)
                    if not md.strip():
                        continue
                    chunks.append(
                        DocumentChunk(
                            chunk_id=self._make_id(filename, page_num, md),
                            source_file=filename,
                            page_number=page_num,
                            chunk_type=ChunkType.TABLE,
                            content=md,
                            metadata={"raw_rows": len(table)},
                        )
                    )
        return chunks

    # ── Image extraction ──────────────────────────────────────────────────────

    def _extract_images(self, pdf_path: Path, filename: str) -> list[DocumentChunk]:
        """Extract images and return IMAGE chunks with a placeholder content.

        The content field is set to a stub here.  The ingestion pipeline
        (src/ingestion/pipeline.py) calls LLMClient.caption_image() and
        replaces the stub with a real GPT-4o caption before embedding.
        """
        chunks: list[DocumentChunk] = []
        doc = fitz.open(str(pdf_path))

        for page_num, page in enumerate(doc, start=1):
            for img_index, img_ref in enumerate(page.get_images(full=True)):
                xref = img_ref[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]

                # Skip tiny decorative images
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                if width * height < MIN_IMAGE_AREA:
                    logger.debug(
                        "Skipping small image on page %d (xref %d): %dx%d",
                        page_num, xref, width, height,
                    )
                    continue

                # Save to cache directory
                img_filename = f"{pdf_path.stem}_p{page_num}_i{img_index}.{ext}"
                img_path = self.image_cache_dir / img_filename
                img_path.write_bytes(image_bytes)

                chunks.append(
                    DocumentChunk(
                        chunk_id=self._make_id(filename, page_num, img_filename),
                        source_file=filename,
                        page_number=page_num,
                        chunk_type=ChunkType.IMAGE,
                        content=f"[IMAGE PENDING CAPTION] {img_filename}",
                        metadata={
                            "image_path": str(img_path),
                            "width": width,
                            "height": height,
                            "ext": ext,
                        },
                    )
                )

        doc.close()
        return chunks

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove excessive whitespace and non-printable characters."""
        text = re.sub(r"\x00", "", text)
        text = re.sub(r" {3,}", "  ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_text(self, text: str) -> list[str]:
        """Simple character-level sliding-window chunker."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    @staticmethod
    def _table_to_markdown(table: list[list]) -> str:
        """Convert a pdfplumber table (list of rows) to Markdown."""
        if not table:
            return ""

        # Sanitise cells
        def clean(cell):
            if cell is None:
                return ""
            return str(cell).replace("\n", " ").strip()

        rows = [[clean(c) for c in row] for row in table]
        header = rows[0]
        separator = ["---"] * len(header)
        body = rows[1:]

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in body:
            # Pad row to header width
            row = row + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    @staticmethod
    def _make_id(filename: str, page_num: int, content: str) -> str:
        """Deterministic chunk ID based on content hash."""
        digest = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{filename}_p{page_num}_{digest}"
