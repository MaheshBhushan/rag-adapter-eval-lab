"""LoRA fine-tuning with a plain PyTorch loop over a Transformers causal LM + PEFT.

The saved adapter (``adapter_config.json`` + ``adapter_model.safetensors``) is the
standard PEFT layout that LoRAX loads with ``adapter_source="local"``.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .data import QAExample
from .prompts import GenRequest, render

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    model_id: str
    output_dir: str
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    epochs: int = 1
    learning_rate: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 4
    warmup_ratio: float = 0.03
    max_length: int = 768
    closed_book_fraction: float = 0.25
    max_steps: int | None = None
    seed: int = 13
    log_every: int = 10


def build_features(tokenizer, examples: list[QAExample], cfg: TrainConfig) -> tuple[list[dict], int]:
    """Tokenise prompt+answer; loss is computed on answer tokens only.

    A ``closed_book_fraction`` of examples drop their passages so the adapter also
    learns the short-answer format when no context is retrieved.
    """
    rng = random.Random(cfg.seed)
    feats, dropped = [], 0
    for ex in examples:
        contexts = [] if rng.random() < cfg.closed_book_fraction else ex.train_contexts
        prompt_ids = tokenizer(render(tokenizer, GenRequest(ex.question, contexts)), add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(ex.answers[0], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        if len(prompt_ids) + len(answer_ids) > cfg.max_length:
            dropped += 1
            continue
        feats.append({"input_ids": prompt_ids + answer_ids, "labels": [-100] * len(prompt_ids) + answer_ids})
    return feats, dropped


def _collate(batch: list[dict], pad_id: int, torch):
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return ids, labels, mask


def train(cfg: TrainConfig, examples: list[QAExample], device: str | None = None) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.float32)
    lora = LoraConfig(r=cfg.r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=cfg.target_modules, task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, lora).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    feats, dropped = build_features(tokenizer, examples, cfg)
    rng = random.Random(cfg.seed)
    steps_per_epoch = math.ceil(len(feats) / (cfg.batch_size * cfg.grad_accum))
    total_steps = cfg.max_steps or steps_per_epoch * cfg.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(optim, int(cfg.warmup_ratio * total_steps), total_steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    losses, step, start = [], 0, time.perf_counter()
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    while step < total_steps:
        order = list(range(len(feats)))
        rng.shuffle(order)
        batches = [order[i : i + cfg.batch_size] for i in range(0, len(order), cfg.batch_size)]
        for start_b in range(0, len(batches), cfg.grad_accum):
            group = batches[start_b : start_b + cfg.grad_accum]
            running = 0.0
            for idx in group:
                ids, labels, mask = (t.to(device) for t in _collate([feats[i] for i in idx], pad_id, torch))
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss / len(group)
                scaler.scale(loss).backward()
                running += loss.item()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            losses.append(running)
            if step % cfg.log_every == 0 or step == total_steps:
                print(f"step {step}/{total_steps} loss {sum(losses[-cfg.log_every:]) / len(losses[-cfg.log_every:]):.4f}", flush=True)
            if step >= total_steps:
                break

    out = Path(cfg.output_dir)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    log = {
        "config": asdict(cfg),
        "device": torch.cuda.get_device_name(0) if use_amp else device,
        "train_examples": len(feats),
        "dropped_too_long": dropped,
        "optimizer_steps": step,
        "trainable_params": trainable,
        "total_params": total,
        "first_10_step_loss": sum(losses[:10]) / max(1, len(losses[:10])),
        "last_10_step_loss": sum(losses[-10:]) / max(1, len(losses[-10:])),
        "train_seconds": round(time.perf_counter() - start, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if use_amp else None,
        "losses": losses,
    }
    (out / "train_log.json").write_text(json.dumps(log, indent=2) + "\n")
    return log
