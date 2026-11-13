"""SQuAD v1.1 exact-match and token F1 (same normalisation as the official script)."""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: list[str]) -> float:
    p = normalize(prediction)
    return float(any(p == normalize(a) for a in answers))


def _f1(prediction: str, truth: str) -> float:
    p, t = normalize(prediction).split(), normalize(truth).split()
    common = Counter(p) & Counter(t)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(t)
    return 2 * precision * recall / (precision + recall)


def f1(prediction: str, answers: list[str]) -> float:
    return max(_f1(prediction, a) for a in answers)


def answer_in_contexts(answers: list[str], contexts: list[str]) -> float:
    """Retrieval hit: does any gold answer string occur in the retrieved text?"""
    joined = normalize(" ".join(contexts))
    return float(any(normalize(a) and normalize(a) in joined for a in answers))
