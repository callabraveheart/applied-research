"""Three ways a pipeline leaks, each isolated so its contribution can be measured.

Every one of these is something I have seen in production code, including my own. None of
them look like mistakes while you are writing them — they look like reasonable
preprocessing. That is what makes them expensive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class GlobalScaler:
    """Leak 1 — normalisation statistics fitted on the full dataset before splitting.

    The mean and variance of every feature encode information from the test period. With
    a heteroskedastic series this matters more than people expect: the scaler tells the
    model, indirectly, which regime each observation belongs to.

    The honest version fits inside each training fold. Both are implemented here so the
    difference can be attributed rather than argued about.
    """

    leak: bool = True

    def __post_init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted_globally = False

    def fit_global(self, X: pd.DataFrame) -> "GlobalScaler":
        if self.leak:
            self._scaler.fit(X.values)
            self._fitted_globally = True
        return self

    def fit_fold(self, X_train: pd.DataFrame) -> "GlobalScaler":
        if not self.leak:
            self._scaler.fit(X_train.values)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.leak and not self._fitted_globally:
            raise RuntimeError("call fit_global first when leak=True")
        return self._scaler.transform(X.values)


def add_lookahead_feature(
    X: pd.DataFrame, raw: pd.DataFrame, horizon: int = 24, strength: float = 0.15
) -> pd.DataFrame:
    """Leak 2 — a feature contaminated with a slice of the future.

    This is the accidental `shift(-1)`, the join against a table that was itself built
    with hindsight, the "as-of" column that was actually restated later. It is rarely
    introduced deliberately; it arrives through a merge.

    `strength` controls how much future information bleeds in, so the resulting apparent
    edge can be traced back to a known quantity instead of guessed at.
    """
    out = X.copy()
    future = (raw["price"].shift(-horizon) / raw["price"] - 1.0).reindex(out.index)

    # Standardise before mixing, or the contamination is swamped by the noise term and
    # the leak silently does nothing. Getting this wrong in the other direction is how a
    # real pipeline hides a leak that only shows up at scale.
    future_z = ((future - future.mean()) / future.std()).fillna(0.0)
    noise = pd.Series(np.random.default_rng(0).standard_normal(len(out)), index=out.index)

    # strength is the target correlation with the future return.
    out["sentiment_score"] = strength * future_z + np.sqrt(1.0 - strength**2) * noise
    return out


def drop_survivors(panel: dict[str, pd.DataFrame], survived: set[str]) -> dict[str, pd.DataFrame]:
    """Leak 3 — survivorship, for the multi-asset extension.

    Selecting the instrument universe using end-of-sample knowledge of which names still
    exist guarantees an upward bias that no amount of careful cross-validation removes,
    because the contamination is in the sampling frame rather than the split.

    Included for completeness and exercised in the multi-asset notebook.
    """
    return {k: v for k, v in panel.items() if k in survived}


LEAK_CATALOGUE = {
    "global_scaling": (
        "Normalisation statistics fitted across the train/test boundary.",
        "Fit every transform inside the training fold. Use a Pipeline so it cannot "
        "be forgotten.",
    ),
    "lookahead_feature": (
        "A feature built from information not available at decision time.",
        "Assert, per column, that the timestamp of the source data is <= t0. Make this "
        "a test, not a code review habit.",
    ),
    "overlapping_labels": (
        "Training labels whose determination window intersects the test window.",
        "Purge overlapping observations and embargo the period after each test fold.",
    ),
    "survivorship": (
        "Universe selected using end-of-sample knowledge of which entities survived.",
        "Reconstruct the universe as it stood at each point in time, including entities "
        "that later disappeared.",
    ),
    "shuffled_split": (
        "Random k-fold applied to time-ordered data, so future rows train the model "
        "that scores past rows.",
        "Use contiguous time blocks. Shuffling is never correct when observations are "
        "serially dependent.",
    ),
}
