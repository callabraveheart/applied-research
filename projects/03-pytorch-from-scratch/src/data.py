"""Character-level corpus and tokenizer.

Downloads tiny-shakespeare when a network is available and falls back to a deterministic
synthetic corpus when it isn't, so CI and a fresh clone both work offline. The fallback
has real structure — a small grammar with long-range agreement — so loss curves on it
mean something rather than measuring memorisation of noise.
"""

from __future__ import annotations

import random
import urllib.error
import urllib.request
from pathlib import Path

import torch

TINY_SHAKESPEARE = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

_NAMES = ["ANTONIO", "BEATRICE", "CORIOLANUS", "DESDEMONA", "EDMUND", "FLORIZEL"]
_VERBS = ["speaks", "answers", "departs", "returns", "kneels", "waits"]
_OBJECTS = ["the letter", "the crown", "his brother", "the morning", "no counsel"]
_CONNECT = ["and yet", "therefore", "but soft", "wherefore", "in truth"]


def _synthetic_corpus(n_lines: int = 12_000, seed: int = 3) -> str:
    """A small grammar with a long-range dependency.

    Whoever is named in a scene heading speaks again at the end of that scene. A model
    with no attention over the full block cannot get that right, so the loss curve is
    measuring something structural rather than character frequencies.
    """
    rng = random.Random(seed)
    lines: list[str] = []
    while len(lines) < n_lines:
        speaker = rng.choice(_NAMES)
        lines.append(f"SCENE. Enter {speaker}.")
        for _ in range(rng.randint(2, 5)):
            lines.append(
                f"{rng.choice(_NAMES)}: {rng.choice(_VERBS)} of {rng.choice(_OBJECTS)}, "
                f"{rng.choice(_CONNECT)}."
            )
        lines.append(f"{speaker}: thus ends the matter.")
        lines.append("")
    return "\n".join(lines)


def load_text(cache: Path = Path("data/corpus.txt")) -> tuple[str, str]:
    """Return (text, source). Source is recorded in results so runs stay comparable."""
    if cache.exists():
        return cache.read_text(encoding="utf-8"), "cached"

    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(TINY_SHAKESPEARE, timeout=15) as resp:
            text = resp.read().decode("utf-8")
        source = "tinyshakespeare"
    except (urllib.error.URLError, TimeoutError, OSError):
        text = _synthetic_corpus()
        source = "synthetic-fallback"

    cache.write_text(text, encoding="utf-8")
    return text, source


class CharTokenizer:
    """The whole tokenizer. Written out because at char level there is no excuse not to."""

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def make_splits(text: str, tokenizer: CharTokenizer, val_frac: float = 0.1):
    """Contiguous split, never random.

    Randomly splitting a character stream puts adjacent windows on both sides of the
    boundary — the same leak as project 01, in a different costume. The held-out text
    here is a contiguous tail the model has never seen any part of.
    """
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(len(data) * (1 - val_frac))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str, generator=None):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    return x.to(device), y.to(device)
