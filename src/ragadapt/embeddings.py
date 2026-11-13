"""Embedding backends shared by both RAG pipelines.

Both the LlamaIndex and the LangChain pipeline wrap the *same* Embedder, so a
comparison between them isolates chunking / indexing / retrieval behaviour.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic offline embedder: L2-normalised hashed bag of word unigrams+bigrams.

    No model download, identical output on every machine; used in tests and CI.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        toks = _TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t).tolist() for t in texts]


class SentenceTransformerEmbedder:
    """Dense embedder (e.g. BAAI/bge-small-en-v1.5) used for the real GPU runs."""

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5", device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device=device)
        self.name = model_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True).tolist()


def make_embedder(spec: str) -> Embedder:
    if spec.startswith("hashing"):
        return HashingEmbedder(int(spec.split("-")[1]) if "-" in spec else 1024)
    return SentenceTransformerEmbedder(spec)
