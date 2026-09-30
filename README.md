# Applied Research

**Most of the cost of applied ML is spent believing results that were never real. These
projects are about the machinery that tells the difference.**

Each project is self-contained, reproducible from a clean clone, and reports what didn't
work alongside what did. Fixed seeds, no data downloads, no API keys required to see the
headline result.

---

## Projects

### [01 — Leakage Audit](projects/01-leakage-audit/)

Data with **zero predictability by construction**. Four validation protocols, identical
model and features. The tutorial default reports a Sharpe of 4.05 with a confidence
interval nowhere near zero.

| Protocol | Sharpe | 95% CI | Contaminated train rows |
|---|---|---|---|
| Shuffled k-fold + global scaling | **4.05** | [3.60, 4.49] | 100% |
| Walk-forward, no purge | 0.23 | [−0.57, 1.01] | 2.2% |
| Purged k-fold + embargo | 0.12 | [−0.64, 0.83] | 0% |
| Purged k-fold + embargo, contaminated feature | **1.28** | [0.64, 1.88] | 0% |

The last row is the point: a verified-clean splitter, and the result is still fake,
because the leak is in a feature. Correct cross-validation is necessary and not
sufficient.

Purging, embargo, and a contamination report that counts overlapping training rows per
fold. Fifteen tests assert the property directly, because a splitter that leaks doesn't
raise — it returns a better number.

### [02 — Agent Evaluation Harness](projects/02-agent-eval-harness/)

Structured extraction from messy commerce listings, evaluated under the rules project 01
argues for. Quality, latency, cost, and a **failure taxonomy** rather than a single
accuracy figure.

Frozen and hashed eval set. Dev/held-out separation enforced by a test, not by
convention — tuning prompts against the evaluation set is project 01's leak arriving
through a different door. Bootstrap confidence intervals on every metric. Runs with no
API key via a deterministic mock provider, so the plumbing is verifiable before anyone
spends a dollar.

### [03 — Transformer from primitives, then made faster](projects/03-pytorch-from-scratch/)

A decoder-only transformer written from matrix operations — no `nn.Transformer`, no
`nn.MultiheadAttention`, no HuggingFace. Trained to 2.09 held-out nats/char from a 4.17
uniform floor, then profiled with every change measured.

**Two of the four optimisations I applied made it slower.** bf16 autocast is a 2.9×
*regression* on a CPU without bf16 hardware support — the cast overhead is paid and none
of the throughput is delivered. That row is the most useful one in the table: the
standard optimisation list is a set of hypotheses about your hardware, not a set of
improvements.

The KV cache is measured across prefix lengths rather than quoted at one: 3.1× at 16
tokens rising to **7.6× at 256**, which is the O(T²) → O(T) claim appearing in wall-clock
time. And `test_kv_cache_matches_full_forward` asserts cached decode is numerically
identical to a full forward pass — a cache bug makes generation faster and subtly wrong,
which is indistinguishable from a successful optimisation unless you check.

---

## The thesis

These look like three different projects. They are one argument in three settings.

An evaluation harness is not scaffolding around research — it *is* the research. The
model is usually the easy part and almost never the thing that's wrong. What's wrong is
the protocol that scored it, and the characteristic failure is silent: it returns a
better number, which is the one result nobody investigates.

So the practice these repos encode is:

1. **Build the evaluation before the model.** Test the splitter. Test the scorer.
2. **Establish a baseline you actually trust,** including a synthetic control where the
   ground truth is known exactly, so a bug is detectable rather than plausible.
3. **Put an interval on everything.** A point estimate with no uncertainty is a claim
   with no content.
4. **Audit results that confirm what you hoped harder than results that don't.**

I did not arrive at this from reading about it. I built a system that produced falsely
profitable results and didn't know for months. The model was fine. The harness was not.

---

## Running anything here

```bash
git clone <repo> && cd applied-research
pip install -r requirements.txt

cd projects/01-leakage-audit && python -m src.experiment && pytest tests/ -q
cd ../02-agent-eval-harness && python -m src.harness && pytest tests/ -q
cd ../03-pytorch-from-scratch && pytest tests/ -q
```

Python 3.11+. Projects 01 and 02 run in under three minutes on a laptop, offline.
CI runs the full test suite on every push.

---

## Background

M.S. Applied and Computational Mathematics, Johns Hopkins (in progress). Six years
building and running systematic trading systems in Python and C#, which is where the
evaluation lessons in this repo were paid for.
