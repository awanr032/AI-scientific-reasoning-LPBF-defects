# Contributing

## Setup

```bash
git clone <repo-url>
cd AI-scientific-reasoning-LPBF-defects
pip install -e ".[notebook,dev]"
```

## Branching

- `main` is always releasable.
- Work on a feature branch (`feat/...`, `fix/...`) and open a PR into `main`.
- Squash-merge preferred, so `main` history stays one commit per change.

## Before opening a PR

```bash
ruff check .
pytest -q
```

Both run in CI (`.github/workflows/ci.yml`) on every push/PR against Python
3.10-3.12; a PR won't merge if either fails.

CI installs only the lightweight dependencies the unit tests need (numpy,
pandas, networkx, faiss-cpu) - it does **not** install torch/transformers/
sentence-transformers, so it can't exercise real embedding or generation
calls. Tests that touch `SemanticIndex` or `Answerer` do so through the fake
embedder/answerer fixtures in `tests/conftest.py`; that's intentional, keep
new tests on that pattern rather than downloading real models in CI. To
actually run the full pipeline against real models, use the Docker image
(`docker build .`) or install the full extras locally.

## Versioning

This package follows [Semantic Versioning](https://semver.org/):
`MAJOR.MINOR.PATCH`.

- **MAJOR**: breaking changes to the public API in `lpbf_defect_reasoning`
  (function signatures, `GraphRagPipeline` behavior, chunk schema
  assumptions).
- **MINOR**: new functionality, backward compatible (e.g. a new retrieval
  strategy, a new CLI flag).
- **PATCH**: bug fixes, doc fixes, dependency bumps that don't change
  behavior.

The version lives in one place: `[project.version]` in `pyproject.toml`.
`lpbf_defect_reasoning.__version__` reads it back via package metadata at
import time - don't hardcode a second copy anywhere.

### "Model" version control

This project does not train its own weights. It composes two pretrained,
third-party models (see `src/lpbf_defect_reasoning/config.py`):

- `sentence-transformers/all-MiniLM-L6-v2` (embeddings)
- `mistralai/Mistral-7B-Instruct-v0.2` (generation)

so there's no checkpoint file to put under version control. Reproducibility
instead comes from **pinning the exact upstream revision** of each model in
`config.py`. Treat a pin bump like any other behavior change:

1. Update `EMBEDDING_MODEL.revision` / `GENERATION_MODEL.revision` in
   `config.py`.
2. Re-run the benchmark (`lpbf_defect_reasoning.evaluation.evaluate_graph_rag`
   against `BENCHMARK`) and note any score change in the PR description.
3. Bump at least the PATCH version and add a `CHANGELOG.md` entry under
   "Changed", even though no application code changed - the pipeline's
   behavior can shift with the upstream weights.

### Releasing

1. Update `version` in `pyproject.toml` and add a section to
   `CHANGELOG.md`.
2. Merge that to `main`.
3. Tag it: `git tag vX.Y.Z && git push origin vX.Y.Z`.
4. `.github/workflows/release.yml` builds the sdist/wheel, builds and pushes
   a Docker image to `ghcr.io/<owner>/<repo>:vX.Y.Z` (and `:latest`), and
   publishes a GitHub Release with both attached.

## Data files

Sample data in `data/sample/` is small (< 6 MB total) and intended as a
runnable demo, not a dataset to grow over time - it's committed as regular
git blobs. If you're adding a genuinely large/growing dataset, don't commit
it directly: track it with [Git LFS](https://git-lfs.github.com/) (see the
commented-out patterns in `.gitattributes` - confirm your git host/CI
credentials support the LFS batch API before enabling it) or, for a real
dataset with its own storage backend, [DVC](https://dvc.org/) with a remote.

## Notebook

`notebooks/AI_Defect_Reasoning_LPBF.ipynb` is a thin demo that imports
`lpbf_defect_reasoning` - it should not contain pipeline logic itself. If
you find yourself writing non-trivial logic in a notebook cell, put it in
`src/lpbf_defect_reasoning/` with a test instead, and call it from the
notebook.
