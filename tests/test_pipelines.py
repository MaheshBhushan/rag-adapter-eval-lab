"""Both RAG pipelines sit behind one interface and answer the same question set."""

import pytest

from ragadapt.embeddings import HashingEmbedder
from ragadapt.generators import ExtractiveGenerator
from ragadapt.harness import evaluate, markdown_table
from ragadapt.pipelines import ClosedBookPipeline, LangChainPipeline, LlamaIndexPipeline, Pipeline


@pytest.fixture(scope="module")
def pipelines(dev_corpus):
    emb = HashingEmbedder()
    built = [LlamaIndexPipeline(emb, top_k=2), LangChainPipeline(emb, top_k=2)]
    for p in built:
        p.build(dev_corpus.documents)
    return built


def test_both_pipelines_share_the_interface(pipelines):
    assert {p.name for p in pipelines} == {"llamaindex", "langchain"}
    assert all(isinstance(p, Pipeline) for p in pipelines)


def test_both_pipelines_answer_the_same_question_set(pipelines, dev_corpus):
    examples = dev_corpus.examples
    gen = ExtractiveGenerator()
    results = [evaluate(f"rag-{p.name}", p, gen, examples) for p in pipelines]
    qids = [[row["qid"] for row in r.predictions] for r in results]
    assert qids[0] == qids[1] == [ex.qid for ex in examples]
    for r in results:
        assert r.n == len(examples)
        assert all(row["prediction"] for row in r.predictions)
        # hashing retrieval over 12 paragraphs should find the gold answer text most of the time
        assert r.retrieval_hit is not None and r.retrieval_hit >= 50.0
        assert r.f1 > 0.0
    table = markdown_table({"runs": [r.summary() for r in results]})
    assert "rag-llamaindex" in table and "rag-langchain" in table


def test_retrieval_returns_gold_paragraph_for_verbatim_query(pipelines, dev_corpus):
    doc = dev_corpus.documents[5]
    for p in pipelines:
        top = p.retrieve(doc.text[:200])
        assert len(top) == 2
        assert doc.text[:100] in top[0]


def test_closed_book_has_no_contexts(dev_corpus):
    r = evaluate("base", ClosedBookPipeline(), ExtractiveGenerator(), dev_corpus.examples[:3])
    assert r.retrieval_hit is None
    assert r.exact_match == 0.0
