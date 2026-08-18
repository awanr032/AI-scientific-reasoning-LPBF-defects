"""End-to-end Graph-RAG pipeline tying retrieval, generation, and reporting together."""

from __future__ import annotations

from typing import Any

import networkx as nx

from .classification import classify_question
from .config import DEFAULT_TOP_K
from .generation import Answerer
from .graph import build_graph
from .indexing import Embedder, SemanticIndex
from .io import load_chunks
from .paths import find_graph_paths
from .reporting import build_structured_report, summarize_evidence
from .retrieval import graph_rag_retrieve


class GraphRagPipeline:
    """Bundles a knowledge graph, semantic index, and answer generator.

    This is the programmatic entry point that replaces the notebook's
    module-level globals (``G``, ``chunks``, ``embedding_model``, ``emb``,
    ``index``, ``model``, ``tokenizer``): construct one instance and reuse it
    across queries instead of relying on notebook cell execution order.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embedder: Embedder,
        answerer: Answerer | None = None,
    ) -> None:
        self.chunks = chunks
        self.graph: nx.DiGraph = build_graph(chunks)
        self.index = SemanticIndex(chunks, embedder)
        self.answerer = answerer

    @classmethod
    def from_chunks_file(
        cls, path: str, embedder: Embedder, answerer: Answerer | None = None
    ) -> GraphRagPipeline:
        return cls(load_chunks(path), embedder=embedder, answerer=answerer)

    def retrieve(self, question: str, k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        return graph_rag_retrieve(question, self.graph, self.index, top_k=k)

    def qa(self, question: str, k: int = DEFAULT_TOP_K) -> tuple[str, list[dict[str, Any]]]:
        """Retrieve evidence and generate a grounded free-text answer."""
        if self.answerer is None:
            raise RuntimeError(
                "GraphRagPipeline was built without an Answerer; pass one in to use qa()."
            )

        evidence = self.retrieve(question, k=k)
        answer = self.answerer.generate(question, evidence)
        return answer, evidence

    def agent(self, question: str, k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        """Run the full agentic pipeline: classify, retrieve, aggregate, answer, report."""
        question_type = classify_question(question)
        evidence = self.retrieve(question, k=k)
        summary = summarize_evidence(evidence)
        paths = find_graph_paths(question, self.graph)

        answer = None
        if question_type != "lookup":
            if self.answerer is None:
                raise RuntimeError(
                    "GraphRagPipeline was built without an Answerer; pass one in to use agent()."
                )
            answer = self.answerer.generate(question, evidence)

        return build_structured_report(
            question=question,
            question_type=question_type,
            summary=summary,
            answer=answer,
            paths=paths,
        )
