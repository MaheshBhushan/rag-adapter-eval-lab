"""Generator adapters: the only place that talks to a language model.

* ``ExtractiveGenerator``  - deterministic offline implementation for tests / CI.
* ``HFGenerator``          - PyTorch + Transformers, optionally with a PEFT LoRA adapter.
* ``LoraxGenerator``       - HTTP client for a LoRAX server serving the same adapter.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from .prompts import GenRequest, clean_answer, render

_WORD = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+")


class Generator(Protocol):
    name: str

    def generate(self, requests: list[GenRequest]) -> list[str]: ...


class ExtractiveGenerator:
    """Returns the context sentence with the highest word overlap with the question.

    No model involved; with no context it answers the empty string. It exists so
    the full harness can run deterministically on CPU without downloads.
    """

    name = "extractive-overlap"

    def generate(self, requests: list[GenRequest]) -> list[str]:
        out = []
        for req in requests:
            q = set(_WORD.findall(req.question.lower()))
            best, best_score = "", 0
            for ctx in req.contexts:
                for sent in _SENT.split(ctx):
                    score = len(q & set(_WORD.findall(sent.lower())))
                    if score > best_score:
                        best, best_score = sent, score
            out.append(best.strip())
        return out


class HFGenerator:
    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
        max_new_tokens: int = 32,
        dtype: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = {"auto": torch.float16 if self.device == "cuda" else torch.float32,
                       "float16": torch.float16, "float32": torch.float32,
                       "bfloat16": torch.bfloat16}[dtype]
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.name = f"hf:{model_id}" + (f"+lora:{adapter_path}" if adapter_path else "")

    def generate(self, requests: list[GenRequest]) -> list[str]:
        prompts = [render(self.tokenizer, r) for r in requests]
        outputs: list[str] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = self.tokenizer(
                prompts[i : i + self.batch_size], return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.device)
            with self.torch.inference_mode():
                gen = self.model.generate(
                    **batch,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new = gen[:, batch["input_ids"].shape[1] :]
            outputs += [clean_answer(t) for t in self.tokenizer.batch_decode(new, skip_special_tokens=True)]
        return outputs


class LoraxGenerator:
    """Sends the exact same rendered prompts to a LoRAX server.

    ``adapter_id`` is a path on the server (``adapter_source="local"``) or a Hub
    repo id (``adapter_source="hub"``); ``None`` queries the base model.
    """

    def __init__(
        self,
        endpoint: str,
        tokenizer_id: str,
        adapter_id: str | None = None,
        adapter_source: str = "local",
        max_new_tokens: int = 32,
        concurrency: int = 16,
        timeout: int = 120,
    ) -> None:
        from lorax import Client
        from transformers import AutoTokenizer

        self.client = Client(endpoint, timeout=timeout)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.adapter_id = adapter_id
        self.adapter_source = adapter_source
        self.max_new_tokens = max_new_tokens
        self.concurrency = concurrency
        self.name = f"lorax:{endpoint}" + (f"+{adapter_id}" if adapter_id else "")

    def _one(self, prompt: str) -> str:
        kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        if self.adapter_id:
            kwargs |= {"adapter_id": self.adapter_id, "adapter_source": self.adapter_source}
        return clean_answer(self.client.generate(prompt, **kwargs).generated_text)

    def generate(self, requests: list[GenRequest]) -> list[str]:
        prompts = [render(self.tokenizer, r) for r in requests]
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(self._one, prompts))
