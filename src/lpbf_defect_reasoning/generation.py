"""LLM-based answer generation grounded in retrieved evidence."""

from __future__ import annotations

from typing import Any, Protocol

EVIDENCE_PROMPT_TEMPLATE = """
Answer the question using only the evidence below.
If evidence is insufficient, say so clearly.

Question:
{question}

Evidence:
{context}

Answer:
"""


def build_prompt(question: str, evidence_chunks: list[dict[str, Any]]) -> str:
    """Assemble the evidence-grounded prompt sent to the generation model."""
    context = "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in evidence_chunks)
    return EVIDENCE_PROMPT_TEMPLATE.format(question=question, context=context)


class Answerer(Protocol):
    """Minimal interface required of an answer generator."""

    def generate(self, question: str, evidence_chunks: list[dict[str, Any]]) -> str:
        ...


class HFCausalLMAnswerer:
    """Answerer backed by a Hugging Face causal LM (e.g. Mistral-7B-Instruct).

    Model/tokenizer loading is lazy and happens in ``__init__`` rather than at
    import time, so importing this module never requires ``transformers`` or
    ``torch`` to be installed (only actually instantiating this class does).
    """

    def __init__(self, model_name: str | None = None, revision: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .config import GENERATION_MODEL

        model_name = model_name or GENERATION_MODEL.repo_id
        revision = revision or GENERATION_MODEL.revision

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate(self, question: str, evidence_chunks: list[dict[str, Any]]) -> str:
        prompt = build_prompt(question, evidence_chunks)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3000
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.2,
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.split("Answer:")[-1].strip()
