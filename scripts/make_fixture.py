"""Cut the small SQuAD v1.1 fixtures in fixtures/ from the official files.

uv run python scripts/make_fixture.py   (downloads into data/raw if missing)
"""

import json
from pathlib import Path

from ragadapt.data import download_squad

ROOT = Path(__file__).resolve().parents[1]


def cut(split: str, articles: int, paragraphs: int, questions: int) -> None:
    raw = json.loads(download_squad(split, ROOT / "data" / "raw").read_text(encoding="utf-8"))
    data = []
    for art in raw["data"][:articles]:
        paras = [{"context": p["context"], "qas": p["qas"][:questions]} for p in art["paragraphs"][:paragraphs]]
        data.append({"title": art["title"], "paragraphs": paras})
    out = ROOT / "fixtures" / f"squad_{split}_mini.json"
    out.write_text(json.dumps({"version": raw["version"], "data": data}, indent=1, ensure_ascii=False) + "\n")
    print(out, sum(len(p["qas"]) for a in data for p in a["paragraphs"]), "questions")


if __name__ == "__main__":
    cut("train", articles=2, paragraphs=4, questions=2)
    cut("dev", articles=3, paragraphs=4, questions=2)
