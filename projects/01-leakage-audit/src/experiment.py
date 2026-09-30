"""Run every protocol against the same data and model, and report the spread.

    python -m src.experiment --n 20000 --horizon 24 --out results.csv

The only thing that varies between rows of the output table is the validation protocol.
Same data, same features, same model, same seed. Any difference in score is therefore
attributable to the protocol and nothing else — which is the entire argument of this
project, and the reason the experiment is built this way rather than as four separate
scripts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .data import assemble
from .leaks import GlobalScaler, add_lookahead_feature
from .validation import LabelSpans, NaiveKFold, PurgedKFold, WalkForward, leakage_report

PERIODS_PER_YEAR = 24 * 365


@dataclass
class Result:
    protocol: str
    leaks: str
    auc: float
    auc_std: float
    sharpe: float
    sharpe_ci_low: float
    sharpe_ci_high: float
    mean_contamination_pct: float
    n_folds: int


def _sharpe(pnl: np.ndarray, horizon: int) -> float:
    """Annualised Sharpe, corrected for overlapping returns.

    Two corrections that are skipped often enough to be worth naming. The forward
    returns overlap by `horizon - 1` periods, so consecutive observations are close to
    duplicates. Annualising with sqrt(periods_per_year) treats every one of them as an
    independent bet and inflates the result by roughly sqrt(horizon) — which is a factor
    of five here, and is on its own enough to turn noise into a publishable number.
    """
    if len(pnl) < 2 or pnl.std(ddof=1) == 0:
        return 0.0
    independent_periods_per_year = PERIODS_PER_YEAR / horizon
    return float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(independent_periods_per_year))


def _pnl(signal: np.ndarray, fwd_ret: np.ndarray) -> np.ndarray:
    return np.where(signal > 0.5, 1.0, -1.0) * fwd_ret


def _block_bootstrap_sharpe_ci(
    pnl: np.ndarray, horizon: int, n_boot: int = 2_000, seed: int = 0
) -> tuple[float, float]:
    """Circular block bootstrap with block length equal to the label horizon.

    An iid bootstrap on overlapping returns produces a confidence interval several times
    too narrow, which is how a Sharpe of 0.3 acquires a t-statistic of 4. Resampling
    whole blocks preserves the dependence the overlap created.
    """
    n = len(pnl)
    if n < 2 * horizon:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / horizon))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(horizon)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1) % n
    samples = pnl[idx[:, :n]]
    sd = samples.std(axis=1, ddof=1)
    scale = np.sqrt(PERIODS_PER_YEAR / horizon)
    sharpes = np.divide(samples.mean(axis=1), sd, out=np.zeros(n_boot), where=sd > 0) * scale
    return float(np.percentile(sharpes, 2.5)), float(np.percentile(sharpes, 97.5))


def run_protocol(
    name: str,
    cv,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    fwd_ret: pd.Series,
    spans: LabelSpans,
    leak_scaling: bool,
    leak_label: str,
    horizon: int,
) -> Result:
    scaler = GlobalScaler(leak=leak_scaling)
    if leak_scaling:
        scaler.fit_global(X)

    aucs: list[float] = []
    all_pred: list[np.ndarray] = []
    all_ret: list[np.ndarray] = []

    for train_idx, test_idx in cv.split(X):
        if len(train_idx) < 100 or len(test_idx) < 20:
            continue
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        if not leak_scaling:
            scaler.fit_fold(X_tr)
        Xtr_s, Xte_s = scaler.transform(X_tr), scaler.transform(X_te)

        model = HistGradientBoostingClassifier(
            max_depth=3, max_iter=150, learning_rate=0.06, random_state=0
        )
        model.fit(Xtr_s, y_tr)
        p = model.predict_proba(Xte_s)[:, 1]

        if len(np.unique(y_te)) > 1:
            aucs.append(roc_auc_score(y_te, p))
        all_pred.append(p)
        all_ret.append(fwd_ret.iloc[test_idx].to_numpy())

    pred = np.concatenate(all_pred) if all_pred else np.array([0.5])
    rets = np.concatenate(all_ret) if all_ret else np.array([0.0])
    pnl = _pnl(pred, rets)
    lo, hi = _block_bootstrap_sharpe_ci(pnl, horizon=horizon)
    contamination = leakage_report(cv, spans, X)

    return Result(
        protocol=name,
        leaks=leak_label,
        auc=float(np.mean(aucs)) if aucs else float("nan"),
        auc_std=float(np.std(aucs)) if aucs else float("nan"),
        sharpe=_sharpe(pnl, horizon=horizon),
        sharpe_ci_low=lo,
        sharpe_ci_high=hi,
        mean_contamination_pct=float(contamination["contamination_pct"].mean()),
        n_folds=len(all_pred),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--embargo", type=float, default=0.01)
    ap.add_argument("--out", type=str, default="results.csv")
    args = ap.parse_args()

    raw, y, X = assemble(n=args.n, horizon=args.horizon)
    fwd_ret = (raw["price"].shift(-args.horizon) / raw["price"] - 1.0).loc[X.index].fillna(0.0)
    spans = LabelSpans.from_horizon(X.index, horizon=args.horizon)
    X_contaminated = add_lookahead_feature(X, raw, horizon=args.horizon)

    purged = lambda: PurgedKFold(  # noqa: E731
        n_splits=args.splits, spans=spans, embargo_pct=args.embargo
    )
    common = dict(y=y, fwd_ret=fwd_ret, spans=spans, horizon=args.horizon)

    results = [
        run_protocol(
            "1. Shuffled k-fold + global scaling",
            NaiveKFold(n_splits=args.splits), X=X,
            leak_scaling=True, leak_label="shuffled_split, global_scaling", **common,
        ),
        run_protocol(
            "2. Walk-forward, no purge",
            WalkForward(n_splits=args.splits), X=X,
            leak_scaling=False, leak_label="overlapping_labels", **common,
        ),
        run_protocol(
            "3. Purged k-fold + embargo",
            purged(), X=X,
            leak_scaling=False, leak_label="none", **common,
        ),
        run_protocol(
            "4. Purged k-fold + embargo, contaminated feature",
            purged(), X=X_contaminated,
            leak_scaling=False, leak_label="lookahead_feature", **common,
        ),
    ]

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(args.out, index=False)

    print("\nGround truth: returns are unpredictable by construction. True Sharpe = 0.\n")
    print(
        df[["protocol", "auc", "sharpe", "sharpe_ci_low", "sharpe_ci_high",
            "mean_contamination_pct"]].to_string(index=False, float_format="%.3f")
    )
    print(f"\nWrote {args.out}")
    print(json.dumps({"n_observations": len(X), "horizon": args.horizon}, indent=2))


if __name__ == "__main__":
    main()
