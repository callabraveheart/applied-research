"""Generate the README figure from results.csv.

    python -m src.figures --results results.csv --out figures/sharpe_by_protocol.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1c1c1c"
CONTAMINATED = "#c0392b"
CLEAN = "#2d7d5a"
MUTED = "#8a8a8a"


def plot(results: pd.DataFrame, out: Path) -> None:
    df = results.iloc[::-1].reset_index(drop=True)
    clean = df["leaks"].eq("none")
    colors = [CLEAN if c else CONTAMINATED for c in clean]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = range(len(df))

    ax.barh(y, df["sharpe"], color=colors, height=0.55, zorder=3)
    ax.errorbar(
        df["sharpe"], y,
        xerr=[df["sharpe"] - df["sharpe_ci_low"], df["sharpe_ci_high"] - df["sharpe"]],
        fmt="none", ecolor=INK, elinewidth=1.4, capsize=4, zorder=4,
    )
    ax.axvline(0.0, color=INK, lw=1.6, zorder=5)
    ax.text(
        0.04, len(df) - 0.35, "true Sharpe = 0\nby construction",
        fontsize=9, color=INK, va="top", style="italic",
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["protocol"], fontsize=9.5)
    ax.set_xlabel("Annualised Sharpe (95% block-bootstrap CI)", fontsize=10)
    ax.set_title(
        "Same data, same model. Only the validation protocol changes.",
        fontsize=12, pad=12, loc="left",
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color=MUTED, alpha=0.25, zorder=0)
    ax.tick_params(left=False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--out", default="figures/sharpe_by_protocol.png")
    args = ap.parse_args()
    plot(pd.read_csv(args.results), Path(args.out))


if __name__ == "__main__":
    main()
