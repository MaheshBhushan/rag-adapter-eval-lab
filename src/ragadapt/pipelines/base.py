"""The one interface every pipeline implements.

A pipeline owns *retrieval* (chunking, indexing, search). Generation is delegated
to a ``Generator`` so the same pipeline can be paired with the base model, a
LoRA-adapted model, or a LoRAX server, and two pipelines paired with the same
generator differ only in how they find context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..data import Document
from ..generators import Generator
from ..prompts import GenRequest


@dataclass
class Prediction:
    question: str
    answer: str
    contexts: list[str]


class Pipeline(ABC):
    name: str

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k

    @abstractmethod
    def build(self, documents: list[Document]) -> None:
        """Index the document collection."""

    @abstractmethod
    def retrieve(self, question: str) -> list[str]:
        """Return up to ``top_k`` context strings for one question."""

    def answer(self, questions: list[str], generator: Generator) -> list[Prediction]:
        contexts = [self.retrieve(q) for q in questions]
        answers = generator.generate([GenRequest(q, c) for q, c in zip(questions, contexts)])
        return [Prediction(q, a, c) for q, a, c in zip(questions, answers, contexts)]


class ClosedBookPipeline(Pipeline):
    """No retrieval: the model answers from its parameters alone."""

    name = "closed-book"

    def build(self, documents: list[Document]) -> None:
        return None

    def retrieve(self, question: str) -> list[str]:
        return []
