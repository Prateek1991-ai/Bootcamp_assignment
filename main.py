"""
Multimodal RAG System — FastAPI Entry Point
Domain: Financial Research & Investment Analysis
Author: BITS WILP Assignment
"""

import time
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.routes import router
from src.retrieval.vector_store import VectorStore
from src.models.llm import LLMClient

# ── Global state ─────────────────────────────────────────────────────────────
START_TIME = time.time()
vector_store: VectorStore = None
llm_client: LLMClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup."""
    global vector_store, llm_client
    vector_store = VectorStore()
    llm_client = LLMClient()

    # Inject into router state
    app.state.vector_store = vector_store
    app.state.llm_client = llm_client
    app.state.start_time = START_TIME

    print("✅ Multimodal RAG system initialised")
    yield
    print("⏹  Shutting down")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multimodal RAG API",
    description=(
        "End-to-end Multimodal Retrieval-Augmented Generation system for "
        "Financial Research & Investment Analysis. Ingests PDFs containing "
        "text, tables, and images; exposes query and management endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
