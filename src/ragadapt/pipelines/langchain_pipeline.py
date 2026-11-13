from __future__ import annotations

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..data import Document
from ..embeddings import Embedder
from .base import Pipeline


class _LangChainEmbeddings(Embeddings):
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed([text])[0]


class LangChainPipeline(Pipeline):
    """RecursiveCharacterTextSplitter -> InMemoryVectorStore -> similarity_search."""

    name = "langchain"

    def __init__(self, embedder: Embedder, top_k: int = 3, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        super().__init__(top_k)
        self.embeddings = _LangChainEmbeddings(embedder)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._store: InMemoryVectorStore | None = None

    def build(self, documents: list[Document]) -> None:
        docs = [LCDocument(page_content=d.text, metadata={"doc_id": d.doc_id, "title": d.title}) for d in documents]
        self._store = InMemoryVectorStore(embedding=self.embeddings)
        self._store.add_documents(self.splitter.split_documents(docs))

    def retrieve(self, question: str) -> list[str]:
        if self._store is None:
            raise RuntimeError("call build() first")
        return [d.page_content for d in self._store.similarity_search(question, k=self.top_k)]
