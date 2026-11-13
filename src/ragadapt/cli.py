"""Command line entry point: prepare -> train -> eval -> report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import (
    download_squad,
    load_examples,
    load_squad_file,
    sample_examples,
    save_examples,
    with_training_contexts,
)

# config name -> (retrieval pipeline, uses LoRA adapter)
CONFIGS = {
    "base": ("closed-book", False),
    "base+rag-llamaindex": ("llamaindex", False),
    "base+rag-langchain": ("langchain", False),
    "lora": ("closed-book", True),
    "lora+rag-llamaindex": ("llamaindex", True),
    "lora+rag-langchain": ("langchain", True),
}


def cmd_prepare(a: argparse.Namespace) -> None:
    out = Path(a.data_dir)
    train_path = Path(a.squad_train) if a.squad_train else download_squad("train", out / "raw")
    dev_path = Path(a.squad_dev) if a.squad_dev else download_squad("dev", out / "raw")
    train_corpus, dev_corpus = load_squad_file(train_path), load_squad_file(dev_path)
    train = with_training_contexts(train_corpus, sample_examples(train_corpus, a.n_train, a.seed), a.distractors, a.seed)
    heldout = sample_examples(dev_corpus, a.n_eval, a.seed)
    save_examples(out / "train.jsonl", train)
    save_examples(out / "heldout.jsonl", heldout)
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for d in dev_corpus.documents:
            fh.write(json.dumps({"doc_id": d.doc_id, "title": d.title, "text": d.text}) + "\n")
    split = {
        "dataset": "SQuAD v1.1",
        "train_source": train_path.name,
        "heldout_source": dev_path.name,
        "train_examples": len(train),
        "heldout_examples": len(heldout),
        "retrieval_corpus_paragraphs": len(dev_corpus.documents),
        "train_articles": len({e.doc_id.split("p")[0] for e in train}),
        "heldout_articles": len({e.doc_id.split("p")[0] for e in heldout}),
        "seed": a.seed,
        "distractors_per_train_example": a.distractors,
    }
    (out / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    print(json.dumps(split, indent=2))


def cmd_train(a: argparse.Namespace) -> None:
    from .train_lora import TrainConfig, train

    cfg = TrainConfig(model_id=a.model_id, output_dir=a.out, r=a.r, lora_alpha=a.alpha, epochs=a.epochs,
                      learning_rate=a.lr, batch_size=a.batch_size, grad_accum=a.grad_accum,
                      max_steps=a.max_steps, max_length=a.max_length, seed=a.seed)
    log = train(cfg, load_examples(Path(a.data_dir) / "train.jsonl"))
    print(json.dumps({k: v for k, v in log.items() if k != "losses"}, indent=2))


def _make_generator(a: argparse.Namespace, adapted: bool):
    from .generators import ExtractiveGenerator, HFGenerator, LoraxGenerator

    if a.backend == "extractive":
        return ExtractiveGenerator()
    if a.backend == "hf":
        return HFGenerator(a.model_id, a.adapter if adapted else None, batch_size=a.batch_size,
                           max_new_tokens=a.max_new_tokens)
    return LoraxGenerator(a.endpoint, a.model_id, a.adapter if adapted else None,
                          max_new_tokens=a.max_new_tokens)


def cmd_eval(a: argparse.Namespace) -> None:
    from .data import Document
    from .embeddings import make_embedder
    from .harness import evaluate, markdown_table, save_results
    from .pipelines import make_pipeline

    data = Path(a.data_dir)
    examples = load_examples(data / "heldout.jsonl")[: a.limit or None]
    with open(data / "corpus.jsonl", encoding="utf-8") as fh:
        documents = [Document(**json.loads(line)) for line in fh]
    configs = a.configs.split(",")
    embedder = make_embedder(a.embedder)
    pipelines = {}
    for name in {CONFIGS[c][0] for c in configs}:
        pipelines[name] = make_pipeline(name, embedder, a.top_k)
        pipelines[name].build(documents)
    generators = {}
    results = []
    for c in configs:
        kind, adapted = CONFIGS[c]
        if adapted not in generators:
            generators[adapted] = _make_generator(a, adapted)
        r = evaluate(c, pipelines[kind], generators[adapted], examples)
        print(json.dumps(r.summary()), flush=True)
        results.append(r)
    meta = {"model_id": a.model_id, "adapter": a.adapter, "backend": a.backend, "embedder": embedder.name,
            "top_k": a.top_k, "max_new_tokens": a.max_new_tokens, "corpus_paragraphs": len(documents)}
    split_file = data / "split.json"
    if split_file.exists():
        meta["split"] = json.loads(split_file.read_text())
    save_results(results, Path(a.out), meta)
    print(markdown_table({"runs": [r.summary() for r in results]}))


def cmd_report(a: argparse.Namespace) -> None:
    from .harness import markdown_table

    print(markdown_table(json.loads(Path(a.summary).read_text())))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ragadapt")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="download SQuAD v1.1 and write the train / held-out split")
    s.add_argument("--data-dir", default="data")
    s.add_argument("--squad-train", help="local SQuAD-format train json (default: download)")
    s.add_argument("--squad-dev", help="local SQuAD-format dev json (default: download)")
    s.add_argument("--n-train", type=int, default=4000)
    s.add_argument("--n-eval", type=int, default=500)
    s.add_argument("--distractors", type=int, default=2)
    s.add_argument("--seed", type=int, default=13)
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("train", help="LoRA fine-tune a causal LM on data/train.jsonl")
    s.add_argument("--data-dir", default="data")
    s.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    s.add_argument("--out", default="adapters/squad-lora")
    s.add_argument("--r", type=int, default=16)
    s.add_argument("--alpha", type=int, default=32)
    s.add_argument("--epochs", type=int, default=1)
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--grad-accum", type=int, default=4)
    s.add_argument("--max-length", type=int, default=768)
    s.add_argument("--max-steps", type=int)
    s.add_argument("--seed", type=int, default=13)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("eval", help="score configurations on the held-out set")
    s.add_argument("--data-dir", default="data")
    s.add_argument("--configs", default=",".join(CONFIGS))
    s.add_argument("--backend", choices=["hf", "lorax", "extractive"], default="hf")
    s.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    s.add_argument("--adapter", help="PEFT adapter dir (hf) or adapter id on the LoRAX server")
    s.add_argument("--endpoint", default="http://127.0.0.1:8080")
    s.add_argument("--embedder", default="BAAI/bge-small-en-v1.5", help="sentence-transformers id or hashing-<dim>")
    s.add_argument("--top-k", type=int, default=3)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--max-new-tokens", type=int, default=32)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--out", default="results/run")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("report", help="print a markdown table from summary.json")
    s.add_argument("summary")
    s.set_defaults(func=cmd_report)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
