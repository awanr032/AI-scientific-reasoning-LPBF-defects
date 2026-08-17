"""Pinned references to the pretrained models this pipeline depends on.

This project does not train its own weights - it *composes* two third-party
pretrained models (a sentence embedder and an instruction-tuned LLM) with a
hand-built knowledge graph. Because there is no custom checkpoint to version,
reproducibility comes from pinning the exact Hugging Face revision of each
upstream model, so a `pip install` today and a `pip install` in six months
resolve to the same weights.

How to update a pin
--------------------
1. Look up the desired model's latest commit SHA:
   ``curl https://huggingface.co/api/models/<repo_id>`` (field ``sha``), or
   the "Files and versions" tab on the model page.
2. Update the ``revision`` below.
3. Bump ``lpbf_defect_reasoning.__version__`` (see ``pyproject.toml``) and
   add an entry to ``CHANGELOG.md`` under "Changed" - a model pin bump is a
   behavior change even though no application code changed.
4. Re-run the benchmark in ``lpbf_defect_reasoning.evaluation`` and record
   the new scores in the PR description.

NOTE: ``revision="main"`` below is a placeholder - this sandbox has no
network access to huggingface.co to resolve a concrete commit SHA. Pin it to
an actual commit hash before relying on this for reproducible deployments.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPin:
    """A pinned reference to a Hugging Face model revision."""

    repo_id: str
    revision: str
    description: str = ""


# Sentence-embedding model used to build the FAISS semantic index.
EMBEDDING_MODEL = ModelPin(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    revision="main",  # TODO: pin to a specific commit SHA before production use
    description="Sentence embedding model for semantic retrieval.",
)

# Instruction-tuned LLM used to generate grounded answers from evidence.
GENERATION_MODEL = ModelPin(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    revision="main",  # TODO: pin to a specific commit SHA before production use
    description="Instruction-tuned LLM for evidence-grounded answer generation.",
)

DEFAULT_TOP_K = 5
MIN_EVIDENCE_SUPPORT = 2  # min. #chunks a label must appear in to count in evaluation
