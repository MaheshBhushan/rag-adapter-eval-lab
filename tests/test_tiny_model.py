"""End-to-end on CPU with a tiny random Qwen2 model: prepare -> LoRA train -> eval all configs."""

import json

from conftest import FIXTURES, TINY_MODEL

from ragadapt.cli import CONFIGS, main


def test_cli_end_to_end_with_tiny_model(tmp_path):
    data, adapter, out = tmp_path / "data", tmp_path / "adapter", tmp_path / "results"
    main(["prepare", "--data-dir", str(data), "--squad-train", str(FIXTURES / "squad_train_mini.json"),
          "--squad-dev", str(FIXTURES / "squad_dev_mini.json"), "--n-train", "16", "--n-eval", "6"])
    split = json.loads((data / "split.json").read_text())
    assert split["train_examples"] == 16 and split["heldout_examples"] == 6

    main(["train", "--data-dir", str(data), "--model-id", TINY_MODEL, "--out", str(adapter),
          "--batch-size", "2", "--grad-accum", "2", "--max-steps", "3"])
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA" and cfg["r"] == 16
    assert (adapter / "adapter_model.safetensors").exists()
    log = json.loads((adapter / "train_log.json").read_text())
    assert log["optimizer_steps"] == 3 and all(x == x for x in log["losses"])  # no NaN

    main(["eval", "--data-dir", str(data), "--model-id", TINY_MODEL, "--adapter", str(adapter),
          "--embedder", "hashing-256", "--top-k", "2", "--max-new-tokens", "4", "--out", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    assert [r["config"] for r in summary["runs"]] == list(CONFIGS)
    assert all(r["n"] == 6 for r in summary["runs"])
    rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    by_config = {}
    for row in rows:
        by_config.setdefault(row["config"], []).append(row["qid"])
    assert len({tuple(v) for v in by_config.values()}) == 1  # identical question set for every config
