"""Tests for the purging and embargo logic.

The point of testing a validation splitter is that its bugs are silent. A splitter that
leaks does not raise — it returns a better number, which is the one failure mode nobody
investigates. These tests assert the property directly: after purging, zero training
label windows intersect the test window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation import LabelSpans, NaiveKFold, PurgedKFold, WalkForward, leakage_report


@pytest.fixture
def spans() -> LabelSpans:
    index = pd.date_range("2020-01-01", periods=1_000, freq="h")
    return LabelSpans.from_horizon(index, horizon=24)


@pytest.fixture
def X(spans: LabelSpans) -> pd.DataFrame:
    return pd.DataFrame({"f": np.arange(len(spans), dtype=float)}, index=spans.t0)


def test_purged_kfold_leaves_no_overlap(spans, X):
    cv = PurgedKFold(n_splits=5, spans=spans, embargo_pct=0.01)
    report = leakage_report(cv, spans, X)
    assert (report["overlapping_train_rows"] == 0).all(), (
        "purging failed — training labels still intersect the test window:\n"
        f"{report}"
    )


def test_naive_kfold_is_contaminated(spans, X):
    """The control condition must actually be contaminated, or the comparison is empty."""
    report = leakage_report(NaiveKFold(n_splits=5), spans, X)
    assert report["overlapping_train_rows"].sum() > 0
    assert report["contamination_pct"].mean() > 50.0


def test_walk_forward_still_leaks_at_the_boundary(spans, X):
    """Respecting time order is not sufficient when labels overlap the split."""
    report = leakage_report(WalkForward(n_splits=5, min_train=200), spans, X)
    assert report["overlapping_train_rows"].sum() > 0


def test_embargo_removes_rows_after_test_window(spans, X):
    no_embargo = PurgedKFold(n_splits=5, spans=spans, embargo_pct=0.0)
    with_embargo = PurgedKFold(n_splits=5, spans=spans, embargo_pct=0.05)
    a = sum(len(tr) for tr, _ in no_embargo.split(X))
    b = sum(len(tr) for tr, _ in with_embargo.split(X))
    assert b < a, "a non-zero embargo must remove training rows"


def test_train_and_test_never_intersect(spans, X):
    for train_idx, test_idx in PurgedKFold(n_splits=4, spans=spans).split(X):
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_every_observation_appears_in_exactly_one_test_fold(spans, X):
    seen = np.concatenate([test for _, test in PurgedKFold(n_splits=5, spans=spans).split(X)])
    assert np.array_equal(np.sort(seen), np.arange(len(X)))


def test_spans_reject_unsorted_input():
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-01"])
    with pytest.raises(ValueError, match="sorted"):
        LabelSpans(t0=index, t1=index)


def test_spans_reject_labels_closing_before_features():
    t0 = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    t1 = pd.DatetimeIndex(["2020-01-01", "2020-01-04"])
    with pytest.raises(ValueError, match="close at or after"):
        LabelSpans(t0=t0, t1=t1)


@pytest.mark.parametrize("horizon", [1, 5, 50, 200])
def test_purging_holds_across_horizons(horizon):
    index = pd.date_range("2020-01-01", periods=2_000, freq="h")
    spans = LabelSpans.from_horizon(index, horizon=horizon)
    X = pd.DataFrame({"f": np.zeros(len(index))}, index=index)
    report = leakage_report(PurgedKFold(n_splits=5, spans=spans), spans, X)
    assert (report["overlapping_train_rows"] == 0).all()


@pytest.mark.parametrize("embargo", [-0.1, 1.0, 1.5])
def test_invalid_embargo_rejected(spans, embargo):
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=5, spans=spans, embargo_pct=embargo)
