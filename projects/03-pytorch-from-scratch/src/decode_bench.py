"""Measure what the KV cache actually buys, and confirm it changes nothing else.

    python -m src.decode_bench --new-tokens 64

Without a cache, each decode step re-attends over the entire prefix: O(T^2) total work to
emit T tokens. With one, each step attends once over cached keys and values: O(T).

The speedup therefore grows with sequence length, which is why measuring it at a single
length is close to useless. This sweeps the prefix length and reports the curve.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import CharTokenizer, load_text
from .model import GPT, GPTConfig


def time_decode(model, prompt, new_tokens: int, *, use_cache: bool, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.manual_seed(0)
        model.generate(prompt, new_tokens, temperature=1.0, use_cache=use_cache)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    # Best-of-N rather than mean: decode latency has a long right tail from scheduler
    # noise, and the question here is what the implementation costs, not what the machine
    # was doing at the time.
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("results/decode.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text, _ = load_text()
    tok = CharTokenizer(text)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
        n_head=4, n_embd=args.n_embd, dropout=0.0, attn_impl="sdpa",
    )
    model = GPT(cfg).to(device).eval()

    rows = []
    for prefix_len in (16, 64, 128, 256):
        prompt = torch.randint(0, tok.vocab_size, (1, prefix_len), device=device)
        no_cache = time_decode(model, prompt, args.new_tokens, use_cache=False)
        cached = time_decode(model, prompt, args.new_tokens, use_cache=True)
        rows.append(
            {
                "prefix_len": prefix_len,
                "no_cache_sec": no_cache,
                "cached_sec": cached,
                "speedup": no_cache / cached,
                "no_cache_ms_per_token": 1000 * no_cache / args.new_tokens,
                "cached_ms_per_token": 1000 * cached / args.new_tokens,
            }
        )
        print(
            f"prefix {prefix_len:>4}: no-cache {no_cache:6.2f}s   cached {cached:6.2f}s"
            f"   {rows[-1]['speedup']:.2f}x"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"device": device, "torch": torch.__version__, "new_tokens": args.new_tokens,
             "model": {"n_layer": args.n_layer, "n_embd": args.n_embd}, "rows": rows},
            indent=2,
        )
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
