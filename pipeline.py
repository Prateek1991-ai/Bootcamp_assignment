"""
Ingestion pipeline.

Orchestrates the full PDF → chunks → embeddings → vector store flow:
  1. Parse PDF (PDFParser) → raw DocumentChunk list
  2. Caption IMAGE chunks   (LLMClient.caption_image)
  3. Embed all chunks       (LLMClient.embed)
  4. Upsert to vector store (VectorStore.add_chunks)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from src.ingestion.parser import PDFParser
from src.models import ChunkType, DocumentChunk, IngestionSummary
from src.models.llm import LLMClient
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Wires together parsing, vision captioning, embedding, and indexing."""

    def __init__(self, llm_client: LLMClient, vector_store: VectorStore) -> None:
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.parser = PDFParser()

    def run(self, pdf_path: Path) -> IngestionSummary:
        """Ingest a PDF end-to-end and return a summary of what was indexed."""
        start = time.perf_counter()
        filename = pdf_path.name

        # ── Step 1: Parse ─────────────────────────────────────────────────────
        raw_chunks = self.parser.parse(pdf_path)

        # ── Step 2: Caption images ────────────────────────────────────────────
        captioned_chunks = self._caption_images(raw_chunks)

        # ── Step 3: Embed ─────────────────────────────────────────────────────
        chunks_with_embeddings = self._embed_chunks(captioned_chunks)

        # ── Step 4: Add to vector store ───────────────────────────────────────
        self.vector_store.add_chunks(chunks_with_embeddings)

        elapsed = time.perf_counter() - start

        text_count = sum(1 for c in captioned_chunks if c.chunk_type == ChunkType.TEXT)
        table_count = sum(1 for c in captioned_chunks if c.chunk_type == ChunkType.TABLE)
        image_count = sum(1 for c in captioned_chunks if c.chunk_type == ChunkType.IMAGE)

        summary = IngestionSummary(
            filename=filename,
            text_chunks=text_count,
            table_chunks=table_count,
            image_chunks=image_count,
            total_chunks=len(captioned_chunks),
            processing_time_seconds=round(elapsed, 2),
        )
        logger.info(
            "Ingested %s in %.1fs — %d text / %d table / %d image",
            filename, elapsed, text_count, table_count, image_count,
        )
        return summary

    # ── Private helpers ───────────────────────────────────────────────────────

    def _caption_images(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Replace placeholder IMAGE chunk content with GPT-4o captions."""
        result: list[DocumentChunk] = []
        for chunk in chunks:
            if chunk.chunk_type != ChunkType.IMAGE:
                result.append(chunk)
                continue

            img_path = Path(chunk.metadata.get("image_path", ""))
            if not img_path.exists():
                logger.warning("Image file not found: %s — skipping", img_path)
                continue

            try:
                caption = self.llm_client.caption_image(img_path)
                chunk = chunk.model_copy(
                    update={
                        "content": f"[Figure on page {chunk.page_number}] {caption}"
                    }
                )
                logger.debug("Captioned %s", img_path.name)
            except Exception as exc:
                logger.error("Failed to caption %s: %s", img_path.name, exc)
                # Keep the chunk with a fallback description
                chunk = chunk.model_copy(
                    update={
                        "content": (
                            f"[Figure on page {chunk.page_number}] "
                            f"Image could not be captioned: {img_path.name}"
                        )
                    }
                )
            result.append(chunk)
        return result

    def _embed_chunks(
        self, chunks: list[DocumentChunk]
    ) -> list[tuple[DocumentChunk, list[float]]]:
        """Embed all chunks in a single batched API call where possible.

        Returns a list of (chunk, embedding_vector) tuples.
        """
        texts = [c.content for c in chunks]

        # Batch size of 100 to stay within OpenAI input limits
        all_embeddings: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.llm_client.embed(batch)
            all_embeddings.extend(embeddings)

        return list(zip(chunks, all_embeddings))
