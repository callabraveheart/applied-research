"""Synthetic price generator with a known ground truth of zero predictability.

Using synthetic data is the point of this project, not a shortcut. On a real series you
can never prove an apparent edge was spurious — someone can always argue the signal is
real and the out-of-sample window was unlucky. Here the data-generating process is
written down, it contains no predictable structure, and the true out-of-sample Sharpe is
exactly zero by construction.

So any protocol that reports an edge on this data is reporting an artifact, and the size
of that artifact is a direct measurement of how much the protocol leaks.

The series is deliberately realistic in the ways that matter for the argument:

  * volatility clustering (GARCH(1,1)) so that scaling statistics vary over time, which
    is what makes global normalisation leak
  * fat tails from the vol process, so naive significance tests overstate confidence
  * zero drift and zero autocorrelation in returns
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GarchParams:
    omega: float = 1.0e-6
    alpha: float = 0.08
    beta: float = 0.90

    def __post_init__(self) -> None:
        if self.alpha + self.beta >= 1.0:
            raise ValueError("alpha + beta must be < 1 for a stationary variance process")


def generate_series(
    n: int = 20_000,
    start: str = "2015-01-01",
    freq: str = "h",
    params: GarchParams | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Return a DataFrame with columns `price`, `ret`, `vol`, indexed by timestamp.

    Returns are unpredictable by construction: r_t = sigma_t * z_t with z iid standard
    normal and sigma_t depending only on past information. E[r_t | F_{t-1}] = 0.
    """
    params = params or GarchParams()
    rng = np.random.default_rng(seed)

    z = rng.standard_normal(n)
    var = np.empty(n)
    ret = np.empty(n)
    var[0] = params.omega / (1.0 - params.alpha - params.beta)
    ret[0] = np.sqrt(var[0]) * z[0]

    for t in range(1, n):
        var[t] = params.omega + params.alpha * ret[t - 1] ** 2 + params.beta * var[t - 1]
        ret[t] = np.sqrt(var[t]) * z[t]

    index = pd.date_range(start=start, periods=n, freq=freq, name="timestamp")
    return pd.DataFrame(
        {"price": 100.0 * np.exp(np.cumsum(ret)), "ret": ret, "vol": np.sqrt(var)},
        index=index,
    )


def build_features(df: pd.DataFrame, windows: tuple[int, ...] = (6, 24, 72)) -> pd.DataFrame:
    """Ordinary momentum / volatility / mean-reversion features.

    Every one of these is computable from information available at time t. None of them
    can predict the next return, because nothing can. If a feature appears to work, the
    protocol is at fault, not the feature.
    """
    out = pd.DataFrame(index=df.index)
    for w in windows:
        out[f"mom_{w}"] = df["ret"].rolling(w).sum()
        out[f"vol_{w}"] = df["ret"].rolling(w).std()
        out[f"zscore_{w}"] = (
            (df["price"] - df["price"].rolling(w).mean()) / df["price"].rolling(w).std()
        )
        out[f"absret_{w}"] = df["ret"].abs().rolling(w).mean()
    out["hour"] = df.index.hour
    out["dow"] = df.index.dayofweek
    return out


def build_labels(df: pd.DataFrame, horizon: int = 24) -> pd.Series:
    """Sign of the forward `horizon`-period return.

    The overlap is the whole issue. Observation t and observation t+1 share horizon-1
    periods of the future. Label windows for nearby observations are almost identical,
    which is exactly the structure purging exists to handle.
    """
    fwd = df["price"].shift(-horizon) / df["price"] - 1.0
    return (fwd > 0).astype(int).rename(f"label_h{horizon}")


def assemble(
    n: int = 20_000, horizon: int = 24, seed: int = 7
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Convenience: raw series, features, labels, aligned and NaN-free."""
    raw = generate_series(n=n, seed=seed)
    X = build_features(raw)
    y = build_labels(raw, horizon=horizon)
    keep = X.notna().all(axis=1) & y.notna()
    keep &= raw.index <= raw.index[-horizon - 1]
    return raw.loc[keep], y.loc[keep], X.loc[keep]
