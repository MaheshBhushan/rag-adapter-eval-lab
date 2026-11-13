"""Shared document-and-question evaluation harness.

Every configuration (closed-book base, RAG via LlamaIndex, RAG via LangChain,
LoRA-adapted, adapted + RAG) is scored on the same examples with the same
metrics by ``evaluate``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .data import QAExample
from .generators import Generator
from .metrics import answer_in_contexts, exact_match, f1
from .pipelines import Pipeline


@dataclass
class RunResult:
    config: str
    pipeline: str
    generator: str
    n: int
    exact_match: float
    f1: float
    retrieval_hit: float | None
    seconds: float
    predictions: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("predictions")
        return d


def evaluate(config: str, pipeline: Pipeline, generator: Generator, examples: list[QAExample]) -> RunResult:
    start = time.perf_counter()
    preds = pipeline.answer([ex.question for ex in examples], generator)
    seconds = time.perf_counter() - start
    rows, em, f, hit = [], 0.0, 0.0, 0.0
    for ex, p in zip(examples, preds, strict=True):
        e, s = exact_match(p.answer, ex.answers), f1(p.answer, ex.answers)
        h = answer_in_contexts(ex.answers, p.contexts) if p.contexts else 0.0
        em, f, hit = em + e, f + s, hit + h
        rows.append({"qid": ex.qid, "question": ex.question, "answers": ex.answers,
                     "prediction": p.answer, "em": e, "f1": s, "retrieval_hit": h})
    n = len(examples)
    retrieves = any(p.contexts for p in preds)
    return RunResult(config, pipeline.name, generator.name, n, 100 * em / n, 100 * f / n,
                     100 * hit / n if retrieves else None, round(seconds, 2), rows)


def save_results(results: list[RunResult], out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "runs": [r.summary() for r in results]}, indent=2) + "\n"
    )
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            for row in r.predictions:
                fh.write(json.dumps({"config": r.config, **row}) + "\n")


def markdown_table(summary: dict) -> str:
    lines = [
        "| Configuration | Retrieval pipeline | n | Exact match | F1 | Answer in retrieved context |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in summary["runs"]:
        hit = "n/a" if r["retrieval_hit"] is None else f"{r['retrieval_hit']:.1f}"
        lines.append(
            f"| {r['config']} | {r['pipeline']} | {r['n']} | {r['exact_match']:.1f} | {r['f1']:.1f} | {hit} |"
        )
    return "\n".join(lines)
