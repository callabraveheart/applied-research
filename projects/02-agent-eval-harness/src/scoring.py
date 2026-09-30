"""Deterministic scoring and a failure taxonomy.

Two decisions worth defending:

**No LLM judge.** A judge model introduces a second system whose errors are correlated
with the system under test, and its agreement with humans is itself unvalidated. For a
task with verifiable ground truth, string comparison after explicit normalisation is
strictly better: it is reproducible, free, and auditable line by line.

**Failures are categorised, not just counted.** An aggregate accuracy of 0.72 could be a
model that is slightly wrong everywhere, or one that is perfect on 90% of items and
returns unparseable output on the rest. Those need different fixes, and the second one is
invisible in a single number.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum


class Failure(str, Enum):
    NONE = "none"
    UNPARSEABLE = "unparseable_output"
    MISSING_FIELD = "missing_field"
    HALLUCINATED_VALUE = "hallucinated_value"
    FORMAT_MISMATCH = "format_mismatch"
    WRONG_VALUE = "wrong_value"
    PROVIDER_ERROR = "provider_error"


@dataclass
class ItemScore:
    item_id: str
    exact_match: bool
    field_accuracy: float
    failures: list[Failure] = field(default_factory=list)
    parsed: dict | None = None

    @property
    def primary_failure(self) -> Failure:
        return self.failures[0] if self.failures else Failure.NONE


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_BARE_OBJECT = re.compile(r"\{.*\}", re.S)


def extract_json(text: str) -> dict | None:
    """Recover an object from model output, tolerating the usual wrappers.

    Being lenient here is a deliberate choice: the question is whether the model got the
    *content* right, not whether it obeyed a formatting instruction. Formatting failures
    are still recorded separately, so leniency does not hide them.
    """
    for candidate in (text, *(m.group(1) for m in _JSON_BLOCK.finditer(text))):
        try:
            value = json.loads(candidate.strip())
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, AttributeError):
            continue
    m = _BARE_OBJECT.search(text)
    if m:
        try:
            value = json.loads(m.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def normalise(value: object) -> str:
    """Casefold, collapse whitespace, strip punctuation that never carries meaning here.

    Every normalisation rule is a judgement call that inflates or deflates the score, so
    they live in one function where they can be read and argued with, rather than being
    scattered through the comparison code.
    """
    s = str(value).strip().casefold()
    s = re.sub(r"[\s_]+", " ", s)
    s = re.sub(r"[^\w\s./-]", "", s)
    return s.strip()


NULLISH = {"", "none", "null", "n/a", "na", "unknown", "not specified", "-"}


def score_item(item_id: str, raw_output: str, truth: dict, error: str | None = None) -> ItemScore:
    if error:
        return ItemScore(item_id, False, 0.0, [Failure.PROVIDER_ERROR])

    parsed = extract_json(raw_output)
    if parsed is None:
        return ItemScore(item_id, False, 0.0, [Failure.UNPARSEABLE])

    failures: list[Failure] = []
    correct = 0

    for key, expected in truth.items():
        # An absent field is correctly represented either by omitting the key or by a
        # null-ish value. Treating omission as an error here was a real bug in the first
        # version of this file: it made "missing_field" the dominant failure mode for
        # every model and buried the differences that actually mattered.
        expected_absent = expected is None or normalise(expected) in NULLISH
        if key not in parsed:
            if expected_absent:
                correct += 1
            else:
                failures.append(Failure.MISSING_FIELD)
            continue

        got, want = normalise(parsed[key]), normalise(expected)
        if expected_absent:
            if got in NULLISH:
                correct += 1
            else:
                failures.append(Failure.HALLUCINATED_VALUE)
            continue

        if got == want:
            correct += 1
        elif got in NULLISH and want not in NULLISH:
            failures.append(Failure.MISSING_FIELD)
        elif want in got or got in want:
            failures.append(Failure.FORMAT_MISMATCH)
        elif got and want and got[0] == want[0] and abs(len(got) - len(want)) <= 3:
            failures.append(Failure.WRONG_VALUE)
        else:
            failures.append(Failure.HALLUCINATED_VALUE)

    accuracy = correct / len(truth) if truth else 0.0
    # Order matters: the taxonomy reports the most actionable failure first.
    priority = [
        Failure.UNPARSEABLE, Failure.MISSING_FIELD,
        Failure.HALLUCINATED_VALUE, Failure.WRONG_VALUE, Failure.FORMAT_MISMATCH,
    ]
    failures.sort(key=lambda f: priority.index(f) if f in priority else 99)
    return ItemScore(item_id, accuracy == 1.0, accuracy, failures, parsed)


def failure_breakdown(scores: list[ItemScore]) -> dict[str, float]:
    """Share of items whose primary failure is each category. Sums to 1."""
    counts = Counter(s.primary_failure.value for s in scores)
    total = max(len(scores), 1)
    return {k: v / total for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}
