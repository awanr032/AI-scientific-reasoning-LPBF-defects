"""FastAPI wrapper exposing the Graph-RAG defect-reasoning pipeline over HTTP.

This turns the same GraphRagPipeline your CLI (cli.py) already uses into a
web service: instead of a human typing a question into a terminal, any other
program can send an HTTP request to /analyze and get the same structured
report back as JSON.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import DEFAULT_TOP_K
from .generation import HFCausalLMAnswerer
from .indexing import load_default_embedder
from .io import load_chunks
from .pipeline import GraphRagPipeline

# Which chunk file this service reasons over. Hardcoded to the sample data
# for now to keep this first version simple - in a real deployment this
# would come from an environment variable instead.
CHUNKS_PATH = "data/sample/graph_rag_chunks.json"

# Off by default: loading the generation LLM (Mistral-7B) takes real time
# and several GB of RAM/VRAM. Keep this False while you're first getting the
# API itself working, same reasoning as --no-generate on the CLI.
ENABLE_GENERATION = False

# Holds the one shared pipeline instance for the life of the running server.
state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the pipeline ONCE when the server starts, not once per request.

    Loading the embedder and building the graph/index is relatively
    expensive - you don't want to redo that work on every single HTTP
    request. This runs once at startup and the same pipeline object is
    reused for every request until the server shuts down.
    """
    chunks = load_chunks(CHUNKS_PATH)
    embedder = load_default_embedder()
    answerer = HFCausalLMAnswerer() if ENABLE_GENERATION else None
    state["pipeline"] = GraphRagPipeline(chunks, embedder=embedder, answerer=answerer)
    yield
    state.clear()


app = FastAPI(
    title="LPBF Defect Reasoning API",
    description="Graph-RAG reasoning over LPBF defect-formation literature.",
    version="0.1.0",
    lifespan=lifespan,
)


class QuestionRequest(BaseModel):
    """Defines exactly what a valid request body must look like.

    FastAPI + Pydantic validate every incoming request against this
    automatically - if 'question' is missing or empty, the caller gets a
    clear 422 error before any of your actual pipeline code even runs.
    """

    question: str = Field(
        ...,
        min_length=1,
        examples=["Which process parameters influence porosity formation?"],
    )
    k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


class ReportResponse(BaseModel):
    """Defines the shape of what this API always returns - mirrors the dict
    that build_structured_report() already produces in reporting.py."""

    question: str
    question_type: str
    top_chunk_ids: list[str]
    defects: dict[str, int]
    parameters: dict[str, int]
    mechanisms: dict[str, int]
    graph_paths: list[list[str]]
    answer: str | None


@app.get("/health")
def health() -> dict[str, str]:
    """Simple liveness check. Cloud platforms/load balancers call this
    endpoint automatically to know whether your service is up."""
    return {"status": "ok"}


@app.post("/analyze", response_model=ReportResponse)
def analyze(request: QuestionRequest) -> dict[str, Any]:
    """The real endpoint: ask a question, get back the structured report.

    Equivalent to what cli.py does, just reachable over HTTP instead of the
    command line.
    """
    pipeline: GraphRagPipeline = state["pipeline"]
    try:
        report = pipeline.agent(request.question, k=request.k)
    except RuntimeError as exc:
        # Same bug you just hit on the CLI (a non-lookup question needs
        # generation, but ENABLE_GENERATION is False) - here it becomes a
        # proper HTTP 400 error with a clear message, instead of crashing
        # the whole server the way the CLI's traceback did.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return report
