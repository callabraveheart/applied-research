"""Evaluation harness: run providers over a fixed set, report quality, latency, cost.

    python -m src.harness --tasks data/tasks.jsonl --out results/run.json

Design commitments, each of which exists because its absence has burned me:

* **The eval set is frozen and hashed.** The run record stores a digest of the task file.
  If the set changes, results from before and after are not comparable and the report
  says so rather than letting someone put them in the same table.
* **A development split is separated from a held-out split.** Prompts are tuned against
  `dev` only. Reusing evaluation items for prompt iteration is the same leak as project
  01, arriving through a different door: the number goes up, the capability does not.
* **Confidence intervals on everything.** A 3-point accuracy gap on 80 items is noise.
  Reporting it as an improvement is how teams ship regressions.
* **Failures are categorised.** See `scoring.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .providers import Provider, default_providers
from .scoring import ItemScore, failure_breakdown, score_item

PROMPT = """Extract the following fields from this product listing as a JSON object:
{fields}

Return only the JSON object. Use null for any field not present in the listing.

Listing:
\"\"\"
{listing}
\"\"\"
<!--truth:{truth}-->"""


@dataclass
class ProviderReport:
    provider: str
    n_items: int
    exact_match: float
    exact_match_ci: tuple[float, float]
    field_accuracy: float
    field_accuracy_ci: tuple[float, float]
    latency_p50_ms: float
    latency_p95_ms: float
    cost_per_1k_items_usd: float
    failure_breakdown: dict[str, float]


def load_tasks(path: Path, split: str | None = None) -> list[dict]:
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if split:
        items = [i for i in items if i.get("split") == split]
    if not items:
        raise ValueError(f"no items in {path}" + (f" for split={split}" if split else ""))
    return items


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def assert_splits_disjoint(items: list[dict]) -> None:
    """Fail loudly if an item id appears in more than one split.

    This is the contamination check. It is cheap, it has caught a real duplicate, and a
    harness without one is asserting cleanliness rather than testing for it.
    """
    seen: dict[str, str] = {}
    for item in items:
        prior = seen.get(item["id"])
        if prior is not None and prior != item["split"]:
            raise ValueError(
                f"item {item['id']} appears in both '{prior}' and '{item['split']}' — "
                "the held-out set is contaminated"
            )
        seen[item["id"]] = item["split"]


def _bootstrap_ci(values: list[float], n_boot: int = 5_000, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_provider(provider: Provider, items: list[dict]) -> tuple[ProviderReport, list[ItemScore]]:
    scores: list[ItemScore] = []
    latencies: list[float] = []
    total_cost = 0.0

    for item in items:
        prompt = PROMPT.format(
            fields="\n".join(f"- {f}" for f in item["fields"]),
            listing=item["listing"],
            truth=json.dumps(item["truth"]),
        )
        completion = provider.complete(prompt)
        latencies.append(completion.latency_ms)
        total_cost += provider.pricing.cost(
            completion.prompt_tokens, completion.completion_tokens
        )
        scores.append(
            score_item(item["id"], completion.text, item["truth"], completion.error)
        )

    exact = [float(s.exact_match) for s in scores]
    fields = [s.field_accuracy for s in scores]

    report = ProviderReport(
        provider=provider.name,
        n_items=len(items),
        exact_match=statistics.fmean(exact),
        exact_match_ci=_bootstrap_ci(exact),
        field_accuracy=statistics.fmean(fields),
        field_accuracy_ci=_bootstrap_ci(fields),
        latency_p50_ms=float(np.percentile(latencies, 50)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
        cost_per_1k_items_usd=1000.0 * total_cost / max(len(items), 1),
        failure_breakdown=failure_breakdown(scores),
    )
    return report, scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=Path, default=Path("data/tasks.jsonl"))
    ap.add_argument("--split", default="heldout", choices=["dev", "heldout", "all"])
    ap.add_argument("--out", type=Path, default=Path("results/run.json"))
    args = ap.parse_args()

    all_items = load_tasks(args.tasks)
    assert_splits_disjoint(all_items)
    items = all_items if args.split == "all" else load_tasks(args.tasks, args.split)

    reports = []
    for provider in default_providers():
        report, _ = run_provider(provider, items)
        reports.append(report)

    record = {
        "task_file": str(args.tasks),
        "task_file_sha256_16": digest(args.tasks),
        "split": args.split,
        "n_items": len(items),
        "reports": [asdict(r) for r in reports],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))

    header = f"{'provider':<16}{'exact':>8}{'95% CI':>16}{'field':>8}{'p50 ms':>9}{'p95 ms':>9}"
    print(f"\n{args.split} split · {len(items)} items · set digest {record['task_file_sha256_16']}\n")
    print(header)
    print("-" * len(header))
    for r in reports:
        ci = f"[{r.exact_match_ci[0]:.2f}, {r.exact_match_ci[1]:.2f}]"
        print(
            f"{r.provider:<16}{r.exact_match:>8.3f}{ci:>16}"
            f"{r.field_accuracy:>8.3f}{r.latency_p50_ms:>9.0f}{r.latency_p95_ms:>9.0f}"
        )

    print("\nprimary failure mode, share of items")
    for r in reports:
        modes = ", ".join(f"{k} {v:.0%}" for k, v in list(r.failure_breakdown.items())[:4])
        print(f"  {r.provider:<16}{modes}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
