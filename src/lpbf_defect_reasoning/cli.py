"""Command-line entry point: ask a question against a chunks file.

Example:
    python -m lpbf_defect_reasoning.cli \\
        --chunks data/sample/graph_rag_chunks.json \\
        --question "Why does high laser power lead to keyhole porosity in LPBF?"
"""

from __future__ import annotations

import argparse
import sys

from .config import DEFAULT_TOP_K
from .generation import HFCausalLMAnswerer
from .indexing import load_default_embedder
from .io import load_chunks
from .pipeline import GraphRagPipeline
from .reporting import pretty_print_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="Path to a chunks JSON file.")
    parser.add_argument("--question", required=True, help="Question to ask.")
    parser.add_argument(
        "--k", type=int, default=DEFAULT_TOP_K, help="Number of evidence chunks to retrieve."
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip loading the generation LLM; only run retrieval + graph reasoning.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    chunks = load_chunks(args.chunks)
    embedder = load_default_embedder()
    answerer = None if args.no_generate else HFCausalLMAnswerer()

    pipeline = GraphRagPipeline(chunks, embedder=embedder, answerer=answerer)
    report = pipeline.agent(args.question, k=args.k)
    pretty_print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
