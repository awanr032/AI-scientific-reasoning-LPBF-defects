# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/) (see `CONTRIBUTING.md` for what
that means for the pinned pretrained models, which have no version of their
own to track otherwise).

## [Unreleased]

## [0.1.0] - 2026-08-17

### Added
- Initial packaged release. Extracted the prototype notebook
  (`AI_Defect_Reasoning_LPBF.ipynb`) into a testable `lpbf_defect_reasoning`
  Python package (`src/lpbf_defect_reasoning/`): chunk loading, graph
  construction, query normalization, graph-guided semantic retrieval,
  LLM-based grounded generation, question classification, graph-path
  discovery, evidence aggregation/reporting, an end-to-end
  `GraphRagPipeline`, benchmark evaluation, and a CLI.
- Pinned upstream model references (`src/lpbf_defect_reasoning/config.py`)
  for `sentence-transformers/all-MiniLM-L6-v2` and
  `mistralai/Mistral-7B-Instruct-v0.2`, since this project has no
  custom-trained checkpoint of its own to version.
- Unit test suite (`tests/`) covering all pure-logic modules against
  synthetic fixtures, with fake embedder/answerer stand-ins so tests never
  require network access or GPU/heavy model downloads.
- GitHub Actions CI (`.github/workflows/ci.yml`): ruff lint + pytest across
  Python 3.10-3.12.
- GitHub Actions release workflow (`.github/workflows/release.yml`),
  triggered on `vX.Y.Z` tags: builds sdist/wheel, builds and pushes a Docker
  image to GHCR, and publishes a GitHub Release.
- `Dockerfile` for a reproducible runtime.
- `.gitattributes` documenting (commented-out) Git LFS patterns for future
  large/binary data additions.
- `pyproject.toml` packaging metadata, pinned `requirements.txt`/
  `requirements-dev.txt`, `CONTRIBUTING.md` versioning/release conventions.

### Changed
- Reorganized sample data into `data/sample/` and moved the demo notebook
  into `notebooks/`, trimmed down to import the package rather than contain
  pipeline logic inline.
