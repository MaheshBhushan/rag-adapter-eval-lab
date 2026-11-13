"""Kaggle GPU kernel: try to serve the trained adapter through LoRAX and evaluate it.

Kaggle kernels have no Docker daemon, so the official image
(ghcr.io/predibase/lorax) is unpacked layer by layer from the registry and its
`lorax-launcher` binary is run directly against the unpacked conda environment.
Every step is logged to /kaggle/working/lorax/attempt.json, whether it works or not.
The adapter comes from the output of the train/eval kernel (kernel_sources).
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

REPO_REF = "__REPO_REF__"
IMAGE, TAG = "predibase/lorax", "main"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = "/tmp/lorax-root"
OUT = "/kaggle/working/lorax"
os.makedirs(OUT, exist_ok=True)
attempt: dict = {"repo_ref": REPO_REF, "image": f"ghcr.io/{IMAGE}:{TAG}", "steps": []}


def log(step: str, **kw) -> None:
    attempt["steps"].append({"step": step, "t": round(time.time()), **kw})
    print(json.dumps(attempt["steps"][-1]), flush=True)
    with open(f"{OUT}/attempt.json", "w") as fh:
        json.dump(attempt, fh, indent=2)


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                f"rag-adapter-eval-lab @ git+https://github.com/MaheshBhushan/rag-adapter-eval-lab@{REPO_REF}",
                "sentence-transformers"], check=True)

import torch  # noqa: E402

log("environment", gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
    whoami=sh("whoami"), docker=sh("which docker || echo 'no docker binary'"),
    disk=sh("df -h /tmp | tail -1"), python=sys.executable)

adapters = glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
adapter_dir = os.path.dirname(adapters[0])
shutil.copytree(adapter_dir, "/tmp/squad-lora", dirs_exist_ok=True)
log("adapter", source=adapter_dir, files=sorted(os.listdir("/tmp/squad-lora")))


def registry(path: str, token: str, accept: str) -> urllib.request.Request:
    return urllib.request.Request(f"https://ghcr.io/v2/{IMAGE}/{path}",
                                  headers={"Authorization": f"Bearer {token}", "Accept": accept})


token = json.load(urllib.request.urlopen(f"https://ghcr.io/token?scope=repository:{IMAGE}:pull"))["token"]
manifest = json.load(urllib.request.urlopen(registry(
    f"manifests/{TAG}", token, "application/vnd.docker.distribution.manifest.v2+json")))
config = json.load(urllib.request.urlopen(registry(f"blobs/{manifest['config']['digest']}", token, "*/*")))
image_env = dict(e.split("=", 1) for e in config["config"].get("Env", []))
log("manifest", layers=len(manifest["layers"]), compressed_gb=round(sum(l["size"] for l in manifest["layers"]) / 1e9, 2),
    entrypoint=config["config"].get("Entrypoint"), cmd=config["config"].get("Cmd"), env=image_env)

t0 = time.time()
os.makedirs(ROOT, exist_ok=True)
for layer in manifest["layers"]:
    with urllib.request.urlopen(registry(f"blobs/{layer['digest']}", token, "*/*")) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for m in tar:
                name = os.path.basename(m.name)
                if name.startswith(".wh."):
                    target = os.path.join(ROOT, os.path.dirname(m.name), name[4:])
                    shutil.rmtree(target, ignore_errors=True) if os.path.isdir(target) else (
                        os.path.lexists(target) and os.remove(target))
                    continue
                if m.isdev():
                    continue
                try:
                    tar.extract(m, ROOT, filter="tar" if hasattr(tarfile, "tar_filter") else None)
                except (OSError, tarfile.TarError):
                    pass
log("unpacked", seconds=round(time.time() - t0), launcher=sh(f"ls -la {ROOT}/usr/local/bin | grep lorax"),
    conda=os.path.exists(f"{ROOT}/opt/conda/bin/python"))

# Scripts inside the image use absolute shebangs (#!/opt/conda/bin/python); point /opt/conda at the unpacked env.
if os.path.exists("/opt/conda") and not os.path.islink("/opt/conda"):
    os.rename("/opt/conda", "/opt/conda.kaggle")
if not os.path.exists("/opt/conda"):
    os.symlink(f"{ROOT}/opt/conda", "/opt/conda")

env = dict(os.environ)
env["PATH"] = f"{ROOT}/usr/local/bin:/opt/conda/bin:" + env["PATH"]
host_libs = ":".join(p for p in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/lib64")
                     if os.path.isdir(p))
env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
    image_env.get("LD_LIBRARY_PATH", "").replace("/usr/", f"{ROOT}/usr/").replace("/opt/conda", "/opt/conda"),
    f"{ROOT}/usr/local/cuda/lib64", "/opt/conda/lib", host_libs]))
env.pop("PYTHONPATH", None)
env["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf"
log("probe_server_import", out=sh(
    f"PATH='{env['PATH']}' LD_LIBRARY_PATH='{env['LD_LIBRARY_PATH']}' /opt/conda/bin/python -c "
    "\"import torch,lorax_server;print(torch.__version__,torch.cuda.is_available())\" 2>&1 | tail -5"))

cmd = ["lorax-launcher", "--model-id", MODEL, "--port", "8080", "--dtype", "float16",
       "--max-input-length", "1536", "--max-total-tokens", "2048", "--max-batch-prefill-tokens", "4096",
       "--adapter-source", "local"]
server_log = open(f"{OUT}/launcher.log", "w")
proc = subprocess.Popen(cmd, env=env, stdout=server_log, stderr=subprocess.STDOUT)
ready = False
for _ in range(180):
    if proc.poll() is not None:
        break
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
        ready = True
        break
    except Exception:
        time.sleep(5)
server_log.flush()
log("launcher", cmd=cmd, ready=ready, exit_code=proc.poll(),
    log_tail=open(f"{OUT}/launcher.log").read()[-6000:])

if ready:
    from ragadapt.cli import main as ragadapt

    data = "/kaggle/input"
    heldout = glob.glob(f"{data}/**/results/heldout.jsonl", recursive=True)[0]
    os.makedirs("/tmp/data", exist_ok=True)
    shutil.copy(heldout, "/tmp/data/heldout.jsonl")
    subprocess.run([sys.executable, "-c", "from ragadapt.data import *; import json;"
                    "c=load_squad_file(download_squad('dev', __import__('pathlib').Path('/tmp/raw')));"
                    "open('/tmp/data/corpus.jsonl','w').writelines(json.dumps(d.__dict__)+'\\n' for d in c.documents)"],
                   check=True)
    ragadapt(["eval", "--data-dir", "/tmp/data", "--backend", "lorax", "--endpoint", "http://127.0.0.1:8080",
              "--model-id", MODEL, "--adapter", "/tmp/squad-lora",
              "--configs", "base,lora,lora+rag-llamaindex", "--embedder", "BAAI/bge-small-en-v1.5",
              "--out", f"{OUT}/results"])
    log("lorax_eval", summary=json.load(open(f"{OUT}/results/summary.json"))["runs"])
    proc.terminate()

attempt["served"] = ready
log("done", served=ready)
