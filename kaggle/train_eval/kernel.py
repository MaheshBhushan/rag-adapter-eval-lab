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
# Kaggle's preinstalled torchao is older than recent PEFT accepts; LoRA does not need it.
run(sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao")
if shutil.which("nvidia-smi"):
    run("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv")
os.makedirs("/tmp/work", exist_ok=True)
os.chdir("/tmp/work")

import peft  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

GPU = torch.cuda.is_available()
env = {"repo_ref": REPO_REF, "device": torch.cuda.get_device_name(0) if GPU else f"cpu x{os.cpu_count()}",
       "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
       "python": sys.version.split()[0]}
for mod in ("llama_index.core", "langchain_core", "sentence_transformers"):
    m = __import__(mod, fromlist=["__version__"])
    env[mod] = getattr(m, "__version__", "?")
print(json.dumps(env, indent=2))

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# Full run on a GPU; a smaller but otherwise identical run when only a CPU session is available.
N_TRAIN, N_EVAL = ("4000", "500") if GPU else ("1000", "200")
run("ragadapt", "prepare", "--data-dir", "data", "--n-train", N_TRAIN, "--n-eval", N_EVAL, "--seed", "13")
run("ragadapt", "train", "--data-dir", "data", "--model-id", MODEL, "--out", "adapters/squad-lora")
run("ragadapt", "eval", "--data-dir", "data", "--model-id", MODEL, "--adapter", "adapters/squad-lora",
    "--embedder", "BAAI/bge-small-en-v1.5", "--top-k", "3", "--batch-size", "32" if GPU else "8",
    "--out", "results")

shutil.copytree("results", f"{OUT}/results", dirs_exist_ok=True)
shutil.copytree("adapters/squad-lora", f"{OUT}/adapter", dirs_exist_ok=True)
shutil.copy("data/split.json", f"{OUT}/results/split.json")
shutil.copy("data/heldout.jsonl", f"{OUT}/results/heldout.jsonl")
with open(f"{OUT}/results/env.json", "w") as fh:
    json.dump(env, fh, indent=2)
print(os.listdir(f"{OUT}/results"))
