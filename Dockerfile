# Reproducible runtime for the LPBF Graph-RAG defect-reasoning pipeline.
#
# Build:
#   docker build -t lpbf-defect-reasoning .
#
# Run against the bundled sample data (mounts nothing extra needed):
#   docker run --rm lpbf-defect-reasoning \
#     --chunks data/sample/graph_rag_chunks.json \
#     --question "Why does high laser power lead to keyhole porosity in LPBF?" \
#     --no-generate
#
# Running with generation enabled (drops --no-generate) downloads the pinned
# Mistral-7B-Instruct model at runtime and needs several GB of RAM/VRAM - see
# src/lpbf_defect_reasoning/config.py for the exact pinned model revisions.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install runtime dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install .

COPY data/sample ./data/sample

ENTRYPOINT ["python", "-m", "lpbf_defect_reasoning.cli"]
CMD ["--help"]
