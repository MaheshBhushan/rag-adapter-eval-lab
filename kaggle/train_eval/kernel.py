"""Kaggle GPU kernel: prepare split -> LoRA fine-tune -> held-out evaluation of every config.

Pushed with `kaggle kernels push -p kaggle/train_eval`; REPO_REF is set by
scripts/kaggle_push.sh to the commit being evaluated.
"""

import json
import os
import shutil
import subprocess
import sys

REPO_REF = "__REPO_REF__"
OUT = "/kaggle/working"


def run(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


run(sys.executable, "-m", "pip", "install", "-q",
    f"rag-adapter-eval-lab[embeddings] @ git+https://github.com/MaheshBhushan/rag-adapter-eval-lab@{REPO_REF}")
run("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv")

import peft  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

env = {"repo_ref": REPO_REF, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "transformers": transformers.__version__, "peft": peft.__version__, "python": sys.version.split()[0]}
for mod in ("llama_index.core", "langchain_core", "sentence_transformers"):
    m = __import__(mod, fromlist=["__version__"])
    env[mod] = getattr(m, "__version__", "?")
print(json.dumps(env, indent=2))

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
run("ragadapt", "prepare", "--data-dir", "data", "--n-train", "4000", "--n-eval", "500", "--seed", "13")
run("ragadapt", "train", "--data-dir", "data", "--model-id", MODEL, "--out", "adapters/squad-lora")
run("ragadapt", "eval", "--data-dir", "data", "--model-id", MODEL, "--adapter", "adapters/squad-lora",
    "--embedder", "BAAI/bge-small-en-v1.5", "--top-k", "3", "--batch-size", "32", "--out", "results")

shutil.copytree("results", f"{OUT}/results", dirs_exist_ok=True)
shutil.copytree("adapters/squad-lora", f"{OUT}/adapter", dirs_exist_ok=True)
shutil.copy("data/split.json", f"{OUT}/results/split.json")
shutil.copy("data/heldout.jsonl", f"{OUT}/results/heldout.jsonl")
with open(f"{OUT}/results/env.json", "w") as fh:
    json.dump(env, fh, indent=2)
print(os.listdir(f"{OUT}/results"))
