from __future__ import annotations

from typing import Any

from llama_index.core import Document as LIDocument
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from pydantic import PrivateAttr

from ..data import Document
from ..embeddings import Embedder
from .base import Pipeline


class _LlamaIndexEmbedding(BaseEmbedding):
    _embedder: Any = PrivateAttr()

    def __init__(self, embedder: Embedder, **kwargs: Any) -> None:
        super().__init__(model_name=embedder.name, embed_batch_size=64, **kwargs)
        self._embedder = embedder

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embedder.embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embedder.embed([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)


class LlamaIndexPipeline(Pipeline):
    """SentenceSplitter -> in-memory VectorStoreIndex -> top-k vector retriever."""

    name = "llamaindex"

    def __init__(self, embedder: Embedder, top_k: int = 3, chunk_size: int = 256, chunk_overlap: int = 32) -> None:
        super().__init__(top_k)
        self.embed_model = _LlamaIndexEmbedding(embedder)
        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._retriever = None

    def build(self, documents: list[Document]) -> None:
        docs = [LIDocument(text=d.text, doc_id=d.doc_id, metadata={"title": d.title}) for d in documents]
        nodes = self.splitter.get_nodes_from_documents(docs)
        index = VectorStoreIndex(nodes, embed_model=self.embed_model)
        self._retriever = index.as_retriever(similarity_top_k=self.top_k)

    def retrieve(self, question: str) -> list[str]:
        if self._retriever is None:
            raise RuntimeError("call build() first")
        return [n.node.get_content() for n in self._retriever.retrieve(question)]
