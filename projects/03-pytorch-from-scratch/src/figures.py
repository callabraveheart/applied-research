"""Loss curve and decode-speedup figures from the results JSON.

    python -m src.figures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1c1c1c"
TRAIN = "#2d5f8a"
VAL = "#c0392b"
MUTED = "#8a8a8a"


def loss_curve(record: dict, out: Path) -> None:
    hist = record["history"]
    steps = [h["step"] for h in hist]
    floor = record.get("uniform_loss_floor") or math.log(record["vocab_size"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})

    ax.plot(steps, [h["train_loss"] for h in hist], color=TRAIN, lw=1.9, label="train")
    ax.plot(steps, [h["val_loss"] for h in hist], color=VAL, lw=1.9, label="held-out")
    ax.axhline(floor, color=MUTED, ls="--", lw=1.2)
    ax.text(
        steps[-1], floor + 0.05, f"uniform floor  ln(V) = {floor:.2f}",
        ha="right", fontsize=8.5, color=MUTED,
    )
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (nats/char)")
    ax.set_title("Training and held-out loss", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)

    gaps = [h["val_loss"] - h["train_loss"] for h in hist]
    ax2.plot(steps, gaps, color=INK, lw=1.7)
    ax2.axhline(0.0, color=MUTED, lw=1.0)
    ax2.set_xlabel("step")
    ax2.set_ylabel("held-out − train")
    ax2.set_title("Generalisation gap", fontsize=11, loc="left")

    for a in (ax, ax2):
        for spine in ("top", "right"):
            a.spines[spine].set_visible(False)
        a.grid(alpha=0.2)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


def decode_curve(record: dict, out: Path) -> None:
    rows = record["rows"]
    x = [r["prefix_len"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.6, 4))
    ax.plot(x, [r["no_cache_ms_per_token"] for r in rows], "o-", color=VAL, lw=1.9,
            label="no cache — O(T²) total")
    ax.plot(x, [r["cached_ms_per_token"] for r in rows], "o-", color=TRAIN, lw=1.9,
            label="KV cache — O(T) total")
    for r in rows:
        ax.annotate(
            f"{r['speedup']:.1f}×", (r["prefix_len"], r["no_cache_ms_per_token"]),
            textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color=INK,
        )
    ax.set_xlabel("prefix length (tokens)")
    ax.set_ylabel("ms per generated token")
    ax.set_title("KV cache: speedup grows with context", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("results/train.json"))
    ap.add_argument("--decode", type=Path, default=Path("results/decode.json"))
    args = ap.parse_args()

    if args.train.exists():
        loss_curve(json.loads(args.train.read_text()), Path("figures/loss_curve.png"))
    if args.decode.exists():
        decode_curve(json.loads(args.decode.read_text()), Path("figures/kv_cache.png"))


if __name__ == "__main__":
    main()
