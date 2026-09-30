"""Leakage-resistant cross-validation for time-ordered data with overlapping labels.

The core problem: when a label at time t is determined by information observed over
[t, t + h], any training observation whose label window overlaps the test window has
seen the test period. Standard k-fold shuffles these together and reports a score that
cannot be achieved out of sample.

Two defences, both from Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7:

  purging   drop training observations whose label window overlaps any test label window
  embargo   additionally drop training observations that begin shortly after a test fold,
            because serial correlation leaks backwards across the boundary

Nothing here is finance-specific. Substitute "user session" for "bar" and the same
argument applies to clickstream, agent trajectories, or any log with overlapping windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


@dataclass(frozen=True)
class LabelSpans:
    """Start and end time of the information window backing each observation.

    Parameters
    ----------
    t0 : pd.DatetimeIndex
        When each observation's feature window closes — the moment a decision is made.
    t1 : pd.DatetimeIndex
        When each observation's label is fully determined. For a 5-bar forward return,
        this is t0 shifted forward 5 bars.
    """

    t0: pd.DatetimeIndex
    t1: pd.DatetimeIndex

    def __post_init__(self) -> None:
        if len(self.t0) != len(self.t1):
            raise ValueError("t0 and t1 must be the same length")
        if not (self.t1 >= self.t0).all():
            raise ValueError("every label must close at or after its feature window")
        if not self.t0.is_monotonic_increasing:
            raise ValueError("observations must be sorted by t0")

    def __len__(self) -> int:
        return len(self.t0)

    @classmethod
    def from_horizon(cls, index: pd.DatetimeIndex, horizon: int) -> "LabelSpans":
        """Build spans for a fixed forward horizon measured in observations."""
        if horizon < 1:
            raise ValueError("horizon must be at least 1 observation")
        end_positions = np.minimum(np.arange(len(index)) + horizon, len(index) - 1)
        return cls(t0=index, t1=index[end_positions])


class PurgedKFold(BaseCrossValidator):
    """K-fold over contiguous time blocks, with purging and an embargo.

    Test folds are contiguous in time. For each fold, a training observation is dropped
    if either of these holds:

      overlap   its label window [t0, t1] intersects the test fold's window
      embargo   its t0 falls in the window immediately after the test fold

    Parameters
    ----------
    n_splits : int
        Number of contiguous test blocks.
    spans : LabelSpans
        Label windows, aligned positionally with X.
    embargo_pct : float
        Embargo length as a fraction of the full sample span. 0.01 is a common default;
        set it to roughly the horizon of any serial correlation you believe is present.
    """

    def __init__(self, n_splits: int, spans: LabelSpans, embargo_pct: float = 0.01) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")
        self.n_splits = int(n_splits)
        self.spans = spans
        self.embargo_pct = float(embargo_pct)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _embargo_delta(self) -> pd.Timedelta:
        total = self.spans.t0[-1] - self.spans.t0[0]
        return pd.Timedelta(total * self.embargo_pct)

    def split(self, X=None, y=None, groups=None):
        n = len(self.spans)
        if X is not None and len(X) != n:
            raise ValueError(f"X has {len(X)} rows but spans describe {n} observations")

        t0, t1 = self.spans.t0, self.spans.t1
        embargo = self._embargo_delta()
        positions = np.arange(n)

        for test_idx in np.array_split(positions, self.n_splits):
            if len(test_idx) == 0:
                continue

            test_start = t0[test_idx[0]]
            test_end = t1[test_idx].max()

            # Purge: any training label window that touches the test window at all.
            overlaps = (t1 >= test_start) & (t0 <= test_end)

            # Embargo: training observations opening just after the test fold closes.
            embargoed = (t0 > test_end) & (t0 <= test_end + embargo)

            train_idx = positions[~(overlaps | embargoed)]
            yield train_idx, test_idx


class NaiveKFold(BaseCrossValidator):
    """Shuffled k-fold — deliberately wrong for time-ordered data.

    Kept in the repo on purpose. It is the control condition: the protocol that produces
    the impressive number. Every result in this project is reported against it.
    """

    def __init__(self, n_splits: int = 5, seed: int = 0) -> None:
        self.n_splits = int(n_splits)
        self.seed = int(seed)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        rng = np.random.default_rng(self.seed)
        positions = rng.permutation(len(X))
        for test_idx in np.array_split(positions, self.n_splits):
            train_idx = np.setdiff1d(positions, test_idx, assume_unique=True)
            yield np.sort(train_idx), np.sort(test_idx)


class WalkForward(BaseCrossValidator):
    """Expanding-window walk-forward with no purging.

    The intermediate condition. Respects time order, which is why practitioners trust it,
    but still leaks whenever labels overlap the split boundary. Included to show that
    "I used walk-forward" is not by itself a defence.
    """

    def __init__(self, n_splits: int = 5, min_train: int = 250) -> None:
        self.n_splits = int(n_splits)
        self.min_train = int(min_train)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        positions = np.arange(n)
        for test_idx in np.array_split(positions[self.min_train:], self.n_splits):
            if len(test_idx) == 0:
                continue
            yield positions[: test_idx[0]], test_idx


def leakage_report(cv: BaseCrossValidator, spans: LabelSpans, X) -> pd.DataFrame:
    """Count, per fold, how many training rows have label windows inside the test window.

    A protocol that is actually clean returns zeros down the `overlapping_train_rows`
    column. Anything else is quantifying its own contamination.
    """
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        test_start = spans.t0[test_idx].min()
        test_end = spans.t1[test_idx].max()
        train_t0, train_t1 = spans.t0[train_idx], spans.t1[train_idx]
        overlapping = int(((train_t1 >= test_start) & (train_t0 <= test_end)).sum())
        rows.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "overlapping_train_rows": overlapping,
                "contamination_pct": 100.0 * overlapping / max(len(train_idx), 1),
            }
        )
    return pd.DataFrame(rows)
