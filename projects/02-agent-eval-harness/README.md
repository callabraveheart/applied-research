# Agent Evaluation Harness

**Structured extraction from messy commerce listings, evaluated the way project 01 says
you have to.**

A small, deterministic harness for comparing models on a fixed task: quality, latency,
cost, and — the part usually missing — *why each one fails*.

Runs immediately with no API key. A mock provider with configurable, realistic failure
modes stands in for real models, so the plumbing is verifiable before anyone spends a
dollar or provisions a credential.

---

## Result

```
heldout split · 120 items · set digest 776f5d9faae8e0ae

provider           exact          95% CI   field   p50 ms   p95 ms
------------------------------------------------------------------
mock-small         0.117    [0.06, 0.17]   0.607      178      259
mock-baseline      0.225    [0.15, 0.30]   0.731      334      502
mock-large         0.300    [0.22, 0.38]   0.814     1290     1925

primary failure mode, share of items
  mock-small      missing_field 44%, format_mismatch 28%, unparseable_output 14%, none 12%
  mock-baseline   missing_field 47%, format_mismatch 24%, none 22%, unparseable_output 6%
  mock-large      missing_field 49%, none 30%, format_mismatch 18%, unparseable_output 2%
```

**These are mock numbers and they are not a model comparison.** The mock reads an answer
key embedded in the prompt; it exercises the harness, not any model's capability. Swap
in real providers by editing `default_providers()` in
[`src/providers.py`](src/providers.py) and rerun.

What the table is for is showing what a comparison ought to contain. Note that
`mock-small` is 7× faster than `mock-large` for 40% of the exact-match rate — that
tradeoff is invisible in an accuracy-only report, and it is usually the decision that
actually gets made.

---

## Four commitments, each with a scar behind it

**The eval set is frozen and hashed.** Every run records a SHA-256 digest of the task
file. Change the set and the digest changes, so results from either side of the change
cannot quietly end up in the same table. Eval sets drift — items get fixed, ambiguous
ones get deleted — and each edit silently moves the baseline.

**Dev and held-out splits are separated and the separation is *tested*.**
`assert_splits_disjoint` fails the run if an item id appears in both. Prompts are tuned
against `dev` only. Iterating prompts against the evaluation set is the same leak as
[project 01](../01-leakage-audit/), arriving through a different door: the number rises,
the capability doesn't. A harness that merely asserts it is clean is not the same as one
that checks.

**Confidence intervals on everything.** `mock-baseline` at 0.225 and `mock-large` at
0.300 have intervals that barely separate on 120 items. On 40 items they would overlap
completely. Shipping a change on a 3-point gap measured over 80 examples is how a
regression gets promoted as an improvement.

**Failures are categorised, not counted.** An aggregate of 0.73 could be a model slightly
wrong everywhere, or one that is perfect on 90% of items and emits unparseable output on
the rest. Those have different fixes, and the second is invisible in a single number.
The taxonomy is in [`src/scoring.py`](src/scoring.py):

| Failure | Meaning | Usually fixed by |
|---|---|---|
| `unparseable_output` | No JSON object recoverable | Constrained decoding, or a stricter output contract |
| `missing_field` | Field present in the listing, absent from output | Prompt, or a required-field retry |
| `hallucinated_value` | Value invented, or supplied for a field that isn't in the listing | The hard one. Usually needs a verification pass |
| `format_mismatch` | Right content, wrong shape | Normalisation, not model choice |
| `wrong_value` | Close but incorrect | Retrieval or context, rarely prompting |
| `provider_error` | Timeout, rate limit, refusal | Infrastructure |

---

## No LLM judge, deliberately

A judge model introduces a second system whose errors correlate with the system under
test, and whose agreement with human raters is itself unvalidated. For a task with
verifiable ground truth, normalised string comparison is strictly better: reproducible,
free, and auditable line by line.

The normalisation rules all live in one function, `normalise()`, because every one of
them is a judgement call that moves the score. They belong somewhere a reviewer can read
and argue with them, not scattered through the comparison code.

---

## A bug this repo keeps on purpose

The first version scored an omitted key as `missing_field` even when the ground truth was
`null`. Since a fifth of the generated listings genuinely lack a field, `missing_field`
became the dominant failure mode for every model and buried the real differences.

The fix is in `score_item` with a comment explaining it. It is left visible rather than
quietly corrected, because that class of bug — a scoring rule that systematically
penalises correct behaviour — is exactly what an evaluation harness exists to make
findable, and it is worth showing the harness catching one.

---

## Running it

```bash
pip install -r requirements.txt
python -m src.make_tasks --n 150          # regenerate the eval set
python -m src.harness --split heldout     # run all providers
pytest tests/ -v
```

Real providers, all with usable free tiers: Groq, Google AI Studio, or a local model
through Ollama's OpenAI-compatible endpoint. `OpenAICompatibleProvider` handles all
three — point `base_url` at the right host and set the key env var.

```python
OpenAICompatibleProvider(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key_env="GROQ_API_KEY",
    pricing=Pricing(input_per_mtok=0.59, output_per_mtok=0.79),
)
```

Check current pricing before quoting a cost.

---

## What I'd build next

- **Retry policy as an evaluated variable.** One retry on unparseable output is nearly
  free and probably removes most of that category. Measure it rather than assuming it.
- **An LLM judge as a *second* grader**, scored against the deterministic one, to report
  where and how far a judge diverges from ground truth.
- **Per-field difficulty.** Aggregate accuracy hides that `material` is much harder than
  `brand`. Field-level breakdowns direct effort.
- **Adversarial slice.** Listings with two brand names, prices in two currencies,
  contradictory sizes. Held-out performance on the easy distribution predicts very little
  about the tail.
