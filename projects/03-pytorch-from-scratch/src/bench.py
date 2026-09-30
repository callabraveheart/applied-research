"""Throughput and memory benchmark across configurations.

    python -m src.bench --steps 12 --out results/bench.json

One variable changes per row. Every row reports tokens/sec and peak memory on the same
model, same batch, same data, same seed, so a difference is attributable.

Three things this does that most benchmark scripts skip, each of which produces a wrong
number when omitted:

* **Warmup steps are discarded.** The first iterations include allocator warmup, cuDNN
  autotuning and — with `torch.compile` — the entire graph compilation. Including them
  makes compile look catastrophic and everything else look slow.
* **CUDA is synchronised before timing.** Kernel launches are asynchronous. Without
  `torch.cuda.synchronize()` you are timing how fast Python can queue work.
* **Median, not mean.** Throughput distributions have a long right tail from scheduler
  noise. The mean reports your worst outliers; the median reports the machine.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .data import CharTokenizer, get_batch, load_text, make_splits
from .model import GPT, GPTConfig


@dataclass
class BenchResult:
    label: str
    tokens_per_sec: float
    step_ms_p50: float
    step_ms_p95: float
    peak_memory_mb: float
    loss_at_end: float
    note: str = ""


def _peak_memory_mb(device: str) -> float:
    """Peak allocated memory, or NaN when the platform cannot attribute it per config.

    On CUDA, `max_memory_allocated` is resettable and gives a clean per-configuration
    number. On CPU the only cheap equivalent is max RSS, which is monotonic over the
    process lifetime — every row after the first would report the high-water mark of all
    previous rows. Reporting that as a per-config measurement would be wrong, so this
    returns NaN instead and the README says why.

    Measuring CPU memory properly means a subprocess per configuration. Worth doing when
    memory is the question; not worth it when throughput is.
    """
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    return float("nan")


def _reset_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def run_config(
    label: str,
    cfg: GPTConfig,
    train_data: torch.Tensor,
    *,
    device: str,
    batch_size: int,
    steps: int,
    warmup: int,
    amp_dtype: torch.dtype | None = None,
    compile_model: bool = False,
    fused_optimizer: bool = False,
    note: str = "",
) -> BenchResult:
    torch.manual_seed(0)
    _reset_memory(device)

    model = GPT(cfg).to(device)
    if compile_model:
        model = torch.compile(model)

    try:
        opt = torch.optim.AdamW(
            model.parameters(), lr=3e-4, fused=fused_optimizer and device == "cuda"
        )
    except (RuntimeError, ValueError) as exc:
        return BenchResult(label, 0, 0, 0, 0, float("nan"), f"unsupported: {exc}")

    gen = torch.Generator().manual_seed(1234)
    autocast = (
        torch.autocast(device_type=device, dtype=amp_dtype)
        if amp_dtype is not None
        else contextlib.nullcontext()
    )

    durations: list[float] = []
    loss_value = float("nan")

    for step in range(steps + warmup):
        x, y = get_batch(train_data, batch_size, cfg.block_size, device, generator=gen)
        _sync(device)
        t0 = time.perf_counter()

        with autocast:
            _, loss, _ = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        _sync(device)
        if step >= warmup:
            durations.append(time.perf_counter() - t0)
        loss_value = float(loss.detach())

    tokens_per_step = batch_size * cfg.block_size
    p50 = statistics.median(durations)
    return BenchResult(
        label=label,
        tokens_per_sec=tokens_per_step / p50,
        step_ms_p50=p50 * 1000,
        step_ms_p95=sorted(durations)[int(0.95 * len(durations)) - 1] * 1000,
        peak_memory_mb=_peak_memory_mb(device),
        loss_at_end=loss_value,
        note=note,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("results/bench.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text, source = load_text()
    tok = CharTokenizer(text)
    train_data, _ = make_splits(text, tok)

    base = dict(
        vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, dropout=0.0,
    )
    common = dict(
        device=device, batch_size=args.batch_size, steps=args.steps, warmup=args.warmup
    )

    configs = [
        ("baseline: manual attention, fp32",
         GPTConfig(**base, attn_impl="manual"), {}),
        ("scaled_dot_product_attention",
         GPTConfig(**base, attn_impl="sdpa"), {}),
        ("SDPA + bf16 autocast",
         GPTConfig(**base, attn_impl="sdpa"), dict(amp_dtype=torch.bfloat16)),
        ("SDPA + bf16 + fused AdamW",
         GPTConfig(**base, attn_impl="sdpa"),
         dict(amp_dtype=torch.bfloat16, fused_optimizer=True)),
        ("SDPA + bf16 + torch.compile",
         GPTConfig(**base, attn_impl="sdpa"),
         dict(amp_dtype=torch.bfloat16, compile_model=True)),
    ]

    results = []
    for label, cfg, kwargs in configs:
        print(f"running: {label} ... ", end="", flush=True)
        try:
            res = run_config(label, cfg, train_data, **common, **kwargs)
        except Exception as exc:  # noqa: BLE001 — a failed config is a result, not a crash
            res = BenchResult(label, 0, 0, 0, 0, float("nan"), f"failed: {type(exc).__name__}")
        results.append(res)
        print(f"{res.tokens_per_sec:,.0f} tok/s" if res.tokens_per_sec else res.note)

    baseline = results[0].tokens_per_sec or float("nan")
    record = {
        "device": device,
        "hardware": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "corpus": source,
        "model_params": {**base},
        "batch_size": args.batch_size,
        "steps": args.steps,
        "results": [asdict(r) for r in results],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))

    print(f"\ndevice={device}  torch={torch.__version__}  threads={torch.get_num_threads()}")
    print(f"{'configuration':<34}{'tok/s':>10}{'vs base':>10}{'p50 ms':>9}")
    print("-" * 63)
    for r in results:
        speedup = f"{r.tokens_per_sec / baseline:.2f}x" if r.tokens_per_sec else "—"
        tps = f"{r.tokens_per_sec:,.0f}" if r.tokens_per_sec else "—"
        print(f"{r.label:<34}{tps:>10}{speedup:>10}{r.step_ms_p50:>9.1f}")
        if r.note:
            print(f"{'':<34}  {r.note}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
