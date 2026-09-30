"""Tests for scoring, JSON recovery, and the contamination check.

A scoring function is the measuring instrument. If it is wrong, every number downstream
is wrong in a way that looks like a result, so it gets tested harder than the code it
measures.
"""

from __future__ import annotations

import json

import pytest

from src.harness import assert_splits_disjoint
from src.scoring import Failure, extract_json, failure_breakdown, normalise, score_item

TRUTH = {"brand": "Kestrel", "color": "moss green", "size": "L", "material": None}


def test_perfect_extraction_is_exact_match():
    out = json.dumps({"brand": "Kestrel", "color": "moss green", "size": "L", "material": None})
    s = score_item("a", out, TRUTH)
    assert s.exact_match
    assert s.field_accuracy == 1.0
    assert s.primary_failure is Failure.NONE


def test_omitting_a_genuinely_absent_field_is_correct():
    """The bug this repo documents: absent ground truth must not count as a miss."""
    out = json.dumps({"brand": "Kestrel", "color": "moss green", "size": "L"})
    assert score_item("a", out, TRUTH).exact_match


def test_inventing_a_value_for_an_absent_field_is_a_hallucination():
    out = json.dumps({**TRUTH, "material": "gore-tex"})
    s = score_item("a", out, TRUTH)
    assert not s.exact_match
    assert Failure.HALLUCINATED_VALUE in s.failures


def test_missing_a_present_field_is_recorded():
    out = json.dumps({"brand": "Kestrel", "color": "moss green", "material": None})
    s = score_item("a", out, TRUTH)
    assert Failure.MISSING_FIELD in s.failures
    assert s.field_accuracy == pytest.approx(3 / 4)


def test_unparseable_output_scores_zero_not_an_exception():
    s = score_item("a", "Sure! Here's the data: {brand: Kestrel,", TRUTH)
    assert s.primary_failure is Failure.UNPARSEABLE
    assert s.field_accuracy == 0.0


def test_provider_error_is_its_own_category():
    s = score_item("a", "", TRUTH, error="429 rate limited")
    assert s.primary_failure is Failure.PROVIDER_ERROR


@pytest.mark.parametrize(
    "wrapper",
    [
        '```json\n{"brand": "Kestrel"}\n```',
        '```\n{"brand": "Kestrel"}\n```',
        'Here you go:\n{"brand": "Kestrel"}\nLet me know if you need more!',
        '{"brand": "Kestrel"}',
    ],
)
def test_json_recovered_from_common_wrappers(wrapper):
    assert extract_json(wrapper) == {"brand": "Kestrel"}


def test_json_extraction_returns_none_rather_than_guessing():
    assert extract_json("no object here at all") is None
    assert extract_json('["a", "list"]') is None


@pytest.mark.parametrize(
    ("a", "b"),
    [("Moss Green", "moss  green"), ("GORE-TEX", "gore-tex"), (" L ", "l"), ("40.50", "40.50")],
)
def test_normalisation_collapses_only_meaningless_differences(a, b):
    assert normalise(a) == normalise(b)


def test_normalisation_preserves_meaningful_differences():
    assert normalise("40.50") != normalise("40.51")
    assert normalise("moss green") != normalise("forest green")


def test_failure_breakdown_sums_to_one():
    scores = [
        score_item("a", json.dumps(TRUTH), TRUTH),
        score_item("b", "garbage", TRUTH),
        score_item("c", json.dumps({"brand": "Kestrel"}), TRUTH),
    ]
    assert sum(failure_breakdown(scores).values()) == pytest.approx(1.0)


def test_contaminated_splits_are_rejected():
    items = [
        {"id": "item-0001", "split": "dev"},
        {"id": "item-0001", "split": "heldout"},
    ]
    with pytest.raises(ValueError, match="contaminated"):
        assert_splits_disjoint(items)


def test_clean_splits_pass():
    assert_splits_disjoint(
        [{"id": "item-0001", "split": "dev"}, {"id": "item-0002", "split": "heldout"}]
    )
