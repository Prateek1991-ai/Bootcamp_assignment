"""
LLM & Embedding client backed by OpenAI.

- Chat completions  → GPT-4o (configurable via OPENAI_LLM_MODEL)
- Vision captioning → GPT-4o with image_url content blocks
- Embeddings        → text-embedding-3-small (configurable)

Retry logic wraps every API call with exponential back-off so transient
rate-limit errors don't crash ingestion.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around the OpenAI client exposing the three capabilities
    the RAG system needs: chat, vision captioning, and embedding."""

    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)
        self.llm_model = settings.openai_llm_model
        self.embedding_model = settings.openai_embedding_model
        logger.info(
            "LLMClient initialised — LLM: %s | Embeddings: %s",
            self.llm_model,
            self.embedding_model,
        )

    # ── Chat completion ───────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def chat(self, system_prompt: str, user_message: str) -> str:
        """Generate a completion from a system + user message pair."""
        response = self._client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    # ── Vision captioning ─────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def caption_image(self, image_path: Path) -> str:
        """Generate a detailed text description of an image for RAG indexing.

        The image is base64-encoded and sent as an image_url block to GPT-4o.
        The resulting caption is what gets embedded — not the raw pixels.
        """
        image_bytes = image_path.read_bytes()
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        # Detect media type from suffix
        suffix = image_path.suffix.lower()
        media_type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(suffix, "image/png")

        response = self._client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "You are analysing a figure extracted from a financial "
                                "research or investment analysis PDF document.\n\n"
                                "Provide a comprehensive description that covers:\n"
                                "1. What type of visual this is (chart, graph, diagram, map, etc.)\n"
                                "2. The key data, trends, or relationships shown\n"
                                "3. Any labels, legends, axes, or numerical values visible\n"
                                "4. The main insight or takeaway an analyst would draw from this figure\n\n"
                                "Be specific and quantitative where possible. "
                                "This description will be used for semantic search retrieval."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=600,
        )
        caption = response.choices[0].message.content.strip()
        logger.debug("Captioned image %s (%d chars)", image_path.name, len(caption))
        return caption

    # ── Embeddings ────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of strings.

        Batches are sent in one API call (OpenAI supports up to 2048 inputs
        per request for text-embedding-3-small).
        """
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_single(self, text: str) -> list[float]:
        """Convenience wrapper for embedding a single string."""
        return self.embed([text])[0]
