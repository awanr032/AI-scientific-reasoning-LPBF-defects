# Reproducible runtime for the LPBF Graph-RAG defect-reasoning pipeline.
#
# Build:
#   docker build -t lpbf-defect-reasoning .
#
# Run as a web API (the default):
#   docker run --rm -p 8000:8000 lpbf-defect-reasoning
#   Then open http://localhost:8000/docs
#
# Run the CLI instead (overrides the default command):
#   docker run --rm lpbf-defect-reasoning \
#     python -m lpbf_defect_reasoning.cli \
#     --chunks data/sample/graph_rag_chunks.json \
#     --question "Which process parameters influence porosity formation?" \
#     --no-generate

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install ".[api]"

COPY data/sample ./data/sample

EXPOSE 8000

CMD uvicorn lpbf_defect_reasoning.api:app --host 0.0.0.0 --port ${PORT:-8000}
