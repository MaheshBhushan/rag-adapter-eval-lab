"""Single source of truth for prompts so every pipeline and generator sees identical text."""

from __future__ import annotations

from dataclasses import dataclass, field

SYSTEM = "You answer questions with a short phrase copied from the facts you know or the given passages. Reply with the answer only."


@dataclass(frozen=True)
class GenRequest:
    question: str
    contexts: list[str] = field(default_factory=list)


def user_message(req: GenRequest) -> str:
    if not req.contexts:
        return f"Question: {req.question}\nAnswer:"
    passages = "\n\n".join(f"Passage {i + 1}: {c}" for i, c in enumerate(req.contexts))
    return f"{passages}\n\nQuestion: {req.question}\nAnswer:"


def chat_messages(req: GenRequest) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(req)}]


def render(tokenizer, req: GenRequest) -> str:
    """Render the chat prompt to plain text (used by HF generation, training and LoRAX)."""
    return tokenizer.apply_chat_template(chat_messages(req), tokenize=False, add_generation_prompt=True)


def clean_answer(text: str) -> str:
    text = text.strip().split("\n")[0].strip()
    for prefix in ("Answer:", "answer:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.rstrip(".").strip()
