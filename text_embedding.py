"""
Inference-only text embedding pipeline for natural-language optimization problems.

This module embeds raw problem descriptions as-is using a pre-trained transformer.
No training, augmentation, or canonicalization is performed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

# Safer defaults for environments where OpenMP/thread runtime can crash.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass
class NLProblem:
    """Single natural-language optimization problem instance."""

    problem_id: str
    text: str
    metadata: Dict[str, str]


class PretrainedTextEmbedder:
    """
    Wrapper around a pre-trained transformer for sentence-level embeddings.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        instruction: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.instruction = instruction
        self.trust_remote_code = trust_remote_code
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if device is None:
            # Keep default deterministic and robust on local laptops.
            device = "cpu"
        self.device = device
        self.model.to(self.device)
        torch.set_num_threads(1)
        self.model.eval()

    @staticmethod
    def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def embed_texts(
        self,
        texts: Sequence[str],
        max_length: int = 256,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Embed raw text descriptions as-is and return normalized vectors [N, H].
        """
        if hasattr(self.model, "encode"):
            # NV-Embed and similar remote-code models typically expose .encode().
            return self._embed_texts_with_model_encode(
                texts=texts,
                max_length=max_length,
                batch_size=batch_size,
            )

        outputs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = list(texts[i : i + batch_size])
                enc = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                model_out = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
                pooled = self.mean_pool(model_out.last_hidden_state, enc["attention_mask"])
                z = F.normalize(pooled, p=2, dim=-1)
                outputs.append(z.cpu())
        if not outputs:
            hidden = self.model.config.hidden_size
            return torch.empty((0, hidden), dtype=torch.float32)
        return torch.cat(outputs, dim=0)

    def _embed_texts_with_model_encode(
        self,
        texts: Sequence[str],
        max_length: int,
        batch_size: int,
    ) -> torch.Tensor:
        outputs = []
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i : i + batch_size])
            vec = None
            attempts = [
                {
                    "instruction": self.instruction if self.instruction is not None else "",
                    "max_length": max_length,
                    "batch_size": len(chunk),
                    "normalize_embeddings": True,
                },
                {
                    "max_length": max_length,
                    "batch_size": len(chunk),
                    "normalize_embeddings": True,
                },
                {},
            ]
            for kwargs in attempts:
                try:
                    vec = self.model.encode(chunk, **kwargs)
                    break
                except TypeError:
                    continue
            if vec is None:
                raise RuntimeError(
                    "Model exposes .encode() but it failed for all tested argument signatures."
                )
            if not isinstance(vec, torch.Tensor):
                vec = torch.tensor(vec, dtype=torch.float32)
            vec = vec.detach().cpu()
            if vec.ndim == 1:
                vec = vec.unsqueeze(0)
            outputs.append(F.normalize(vec, p=2, dim=-1))
        if not outputs:
            hidden = getattr(self.model.config, "hidden_size", 0)
            return torch.empty((0, hidden), dtype=torch.float32)
        return torch.cat(outputs, dim=0)

    def embed_problems(
        self,
        problems: Sequence[NLProblem],
        max_length: int = 256,
        batch_size: int = 32,
    ) -> torch.Tensor:
        return self.embed_texts(
            texts=[p.text for p in problems],
            max_length=max_length,
            batch_size=batch_size,
        )


def embed_texts_pretrained(
    texts: Sequence[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_length: int = 256,
    batch_size: int = 32,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    instruction: Optional[str] = None,
) -> torch.Tensor:
    """
    Convenience function for one-shot embedding.
    """
    embedder = PretrainedTextEmbedder(
        model_name=model_name,
        device=device,
        trust_remote_code=trust_remote_code,
        instruction=instruction,
    )
    return embedder.embed_texts(texts=texts, max_length=max_length, batch_size=batch_size)


if __name__ == "__main__":
    sample_text = (
        "An airport has two available gates (G1, G2) and needs to assign four flights "
        "(F1: 10:00-11:00, F2: 10:30-11:30, F3: 11:15-12:30, F4: 10:45-11:45) to these gates. "
        "Flights assigned to the same gate must have non-overlapping time intervals, and the goal "
        "is to minimize flight delays, where each 1-minute delay beyond the scheduled departure "
        "time incurs a cost of 1 unit. How should the flights be assigned to the gates to achieve "
        "this objective?"
    )
    texts = [
        sample_text,
        "Assign crews to flights with legal rest constraints and shift limits while minimizing overtime costs.",
    ]
    Z_text = embed_texts_pretrained(
        texts=texts,
    )
    print(Z_text.shape)
