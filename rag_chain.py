"""
RAG chain: retrieval → prompt construction → answer generation.

The chain:
  1. Embeds the user question via OpenAI embeddings
  2. Retrieves the top-k most relevant chunks from FAISS
  3. Builds a grounded prompt that includes the retrieved context
  4. Calls GPT-4o to generate a factual, cited answer
  5. Returns the answer alongside source references

The prompt is intentionally conservative — the model is instructed to
answer ONLY from the provided context and to say "I don't know" if
the context is insufficient, preventing hallucination.
"""

from __future__ import annotations

import logging

from src.models import ChunkType, DocumentChunk, QueryRequest, QueryResponse, SourceReference
from src.models.llm import LLMClient
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert financial research analyst assistant.
Your job is to answer questions about financial documents, investment reports,
earnings disclosures, and market analyses using ONLY the context provided below.

Rules:
1. Answer ONLY based on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question,
   say "The provided documents do not contain sufficient information to answer this question."
3. When referencing specific data (numbers, percentages, dates), be precise.
4. If multiple sources support your answer, synthesise them coherently.
5. Always cite which document and page your answer is drawn from.
6. For table-based answers, present the relevant data clearly.
7. For image-based answers, describe what the figure shows and what it implies.

Format:
- Provide a clear, concise answer (2–5 paragraphs maximum).
- End with a "Sources:" section listing the document names and pages used.
"""


class RAGChain:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(self, llm_client: LLMClient, vector_store: VectorStore) -> None:
        self.llm_client = llm_client
        self.vector_store = vector_store

    def query(self, request: QueryRequest) -> QueryResponse:
        """Execute a full RAG query and return a structured response."""
        question = request.question
        top_k = request.top_k or None

        # ── Embed the question ────────────────────────────────────────────────
        query_embedding = self.llm_client.embed_single(question)

        # ── Retrieve relevant chunks ──────────────────────────────────────────
        chunks = self.vector_store.search(query_embedding, top_k=top_k)

        if not chunks:
            return QueryResponse(
                question=question,
                answer="No documents have been indexed yet. Please ingest a PDF first via POST /ingest.",
                sources=[],
                chunks_retrieved=0,
                model_used=self.llm_client.llm_model,
            )

        # ── Build context string ──────────────────────────────────────────────
        context_blocks = self._build_context(chunks)

        # ── Generate answer ───────────────────────────────────────────────────
        user_message = (
            f"Context:\n{context_blocks}\n\n"
            f"Question: {question}\n\n"
            "Answer (based solely on the context above):"
        )
        answer = self.llm_client.chat(SYSTEM_PROMPT, user_message)

        # ── Build source references ───────────────────────────────────────────
        sources = [
            SourceReference(
                chunk_id=c.chunk_id,
                source_file=c.source_file,
                page_number=c.page_number,
                chunk_type=c.chunk_type,
                content_preview=c.content[:200],
            )
            for c in chunks
        ]

        logger.info(
            "Query answered — %d chunks retrieved, answer length %d chars",
            len(chunks), len(answer),
        )
        return QueryResponse(
            question=question,
            answer=answer,
            sources=sources,
            chunks_retrieved=len(chunks),
            model_used=self.llm_client.llm_model,
        )

    # ── Context builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: list[DocumentChunk]) -> str:
        """Format retrieved chunks into a single context string for the prompt."""
        blocks: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            type_label = {
                ChunkType.TEXT: "Text",
                ChunkType.TABLE: "Table",
                ChunkType.IMAGE: "Figure",
            }.get(chunk.chunk_type, "Content")

            block = (
                f"[Source {i}] {type_label} from '{chunk.source_file}' (page {chunk.page_number})\n"
                f"{chunk.content}"
            )
            blocks.append(block)

        return "\n\n---\n\n".join(blocks)
