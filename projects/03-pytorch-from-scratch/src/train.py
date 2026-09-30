"""Training loop with held-out evaluation and a loss curve.

    python -m src.train --steps 1200 --eval-interval 100

Deliberately plain. The interesting engineering is in `bench.py`; this exists to show the
model learns, and to produce the curve and the samples that make that claim checkable.

The one thing worth arguing about: validation loss is computed on a **contiguous tail**
of the corpus, never a random split. Randomly splitting a character stream puts adjacent
windows on both sides of the boundary, which is project 01's leak wearing a different
costume. It would make the validation curve track the training curve beautifully and mean
nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .data import CharTokenizer, get_batch, load_text, make_splits
from .model import GPT, GPTConfig


@torch.no_grad()
def estimate_loss(model, data, cfg, batch_size, device, iters=20, seed=0):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = get_batch(data, batch_size, cfg.block_size, device, generator=gen)
        _, loss, _ = model(x, targets=y)
        losses[i] = loss.item()
    model.train()
    return float(losses.mean())


def cosine_lr(step: int, *, warmup: int, total: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    if step > total:
        return min_lr
    ratio = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--attn", default="sdpa", choices=["manual", "sdpa"])
    ap.add_argument("--out", type=Path, default=Path("results/train.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)

    text, source = load_text()
    tok = CharTokenizer(text)
    train_data, val_data = make_splits(text, tok)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, dropout=0.1, attn_impl=args.attn,
    )
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))

    print(f"device={device}  corpus={source}  vocab={tok.vocab_size}")
    print(f"params={model.n_params():,}  train={len(train_data):,}  val={len(val_data):,}")
    # A char-level model that cannot beat uniform over the vocabulary has not learned
    # anything. Stating the floor up front stops a loss of 3.2 looking like progress.
    print(f"uniform-prediction loss floor = ln({tok.vocab_size}) = {math.log(tok.vocab_size):.3f}\n")

    gen = torch.Generator().manual_seed(42)
    history: list[dict] = []
    start = time.perf_counter()

    for step in range(args.steps + 1):
        lr = cosine_lr(step, warmup=100, total=args.steps, lr=args.lr, min_lr=args.lr / 10)
        for group in opt.param_groups:
            group["lr"] = lr

        if step % args.eval_interval == 0:
            tr = estimate_loss(model, train_data, cfg, args.batch_size, device)
            va = estimate_loss(model, val_data, cfg, args.batch_size, device)
            history.append({"step": step, "train_loss": tr, "val_loss": va, "lr": lr})
            print(
                f"step {step:>5}  train {tr:.4f}  val {va:.4f}  gap {va - tr:+.4f}  lr {lr:.2e}"
            )

        x, y = get_batch(train_data, args.batch_size, cfg.block_size, device, generator=gen)
        _, loss, _ = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    elapsed = time.perf_counter() - start
    prompt = torch.tensor([tok.encode("\n")], dtype=torch.long, device=device)
    sample = tok.decode(model.generate(prompt, 400, temperature=0.8, top_k=40)[0].tolist())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "device": device, "corpus": source, "vocab_size": tok.vocab_size,
                "params": model.n_params(), "uniform_loss_floor": math.log(tok.vocab_size),
                "config": vars(args) | {"out": str(args.out)},
                "elapsed_sec": elapsed, "history": history, "sample": sample,
            },
            indent=2, default=str,
        )
    )
    print(f"\ntrained in {elapsed / 60:.1f} min\n\n--- sample ---\n{sample}\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
