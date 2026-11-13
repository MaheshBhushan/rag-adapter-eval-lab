"""SQuAD v1.1 loading and the train / held-out split used by every experiment.

SQuAD v1.1 ships a train file and a dev file whose Wikipedia articles are
disjoint, so the dev file is a clean held-out set: no held-out paragraph or
question is ever seen during LoRA training.
"""

from __future__ import annotations

import json
import random
import shutil
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

SQUAD_URLS = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class QAExample:
    qid: str
    question: str
    answers: list[str]
    doc_id: str
    # Contexts used only for supervised LoRA training (gold paragraph + distractors).
    train_contexts: list[str] = field(default_factory=list)


@dataclass
class Corpus:
    documents: list[Document]
    examples: list[QAExample]

    def doc(self, doc_id: str) -> Document:
        return self._by_id[doc_id]

    def __post_init__(self) -> None:
        self._by_id = {d.doc_id: d for d in self.documents}


def download_squad(split: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / Path(SQUAD_URLS[split]).name
    if not path.exists():
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(SQUAD_URLS[split], timeout=60) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        tmp.rename(path)
    return path


def parse_squad(raw: dict) -> Corpus:
    documents: list[Document] = []
    examples: list[QAExample] = []
    for a_idx, article in enumerate(raw["data"]):
        for p_idx, para in enumerate(article["paragraphs"]):
            doc_id = f"a{a_idx}p{p_idx}"
            documents.append(Document(doc_id, article["title"], para["context"]))
            for qa in para["qas"]:
                answers = list(dict.fromkeys(a["text"] for a in qa["answers"]))
                examples.append(QAExample(qa["id"], qa["question"].strip(), answers, doc_id))
    return Corpus(documents, examples)


def load_squad_file(path: Path) -> Corpus:
    return parse_squad(json.loads(Path(path).read_text(encoding="utf-8")))


def sample_examples(corpus: Corpus, n: int, seed: int) -> list[QAExample]:
    rng = random.Random(seed)
    if n >= len(corpus.examples):
        return list(corpus.examples)
    return rng.sample(corpus.examples, n)


def with_training_contexts(
    corpus: Corpus, examples: list[QAExample], n_distractors: int, seed: int
) -> list[QAExample]:
    """Attach gold paragraph + distractor paragraphs from the same article.

    This mimics what a retriever returns at inference time, so the adapter learns
    to pick the answer span out of several passages rather than from one.
    """
    rng = random.Random(seed)
    by_article: dict[str, list[str]] = {}
    for d in corpus.documents:
        by_article.setdefault(d.doc_id.split("p")[0], []).append(d.doc_id)
    out = []
    for ex in examples:
        pool = [d for d in by_article[ex.doc_id.split("p")[0]] if d != ex.doc_id]
        picks = rng.sample(pool, min(n_distractors, len(pool))) + [ex.doc_id]
        rng.shuffle(picks)
        out.append(
            QAExample(ex.qid, ex.question, ex.answers, ex.doc_id, [corpus.doc(d).text for d in picks])
        )
    return out


def save_examples(path: Path, examples: list[QAExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")


def load_examples(path: Path) -> list[QAExample]:
    with open(path, encoding="utf-8") as f:
        return [QAExample(**json.loads(line)) for line in f if line.strip()]
