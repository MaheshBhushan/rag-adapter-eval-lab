from .base import ClosedBookPipeline, Pipeline, Prediction
from .langchain_pipeline import LangChainPipeline
from .llamaindex_pipeline import LlamaIndexPipeline

__all__ = ["ClosedBookPipeline", "LangChainPipeline", "LlamaIndexPipeline", "Pipeline", "Prediction"]


def make_pipeline(kind: str, embedder=None, top_k: int = 3) -> Pipeline:
    if kind == "closed-book":
        return ClosedBookPipeline(top_k)
    if kind == "llamaindex":
        return LlamaIndexPipeline(embedder, top_k)
    if kind == "langchain":
        return LangChainPipeline(embedder, top_k)
    raise ValueError(f"unknown pipeline {kind!r}")
