# Research log

Dated notes on what was tried and what it cost. Kept because the interesting part of a
result is usually the three versions that came before it, and those are gone by the time
anyone writes a README.

---

## 2026-09-25 — leakage audit working end to end

Built the synthetic generator, the four protocols, and the contamination report.

**First run was wrong in two ways and both were flattering.**

The shuffled-k-fold protocol reported a Sharpe of **26.6**. That is absurd on its face,
and the absurdity is what made it findable — a Sharpe of 2.6 would have gone straight
into the README. Two causes:

1. **Annualisation ignored the overlap.** 24-period forward returns sampled hourly
   overlap by 23 periods. Annualising with `sqrt(24 * 365)` treats every observation as
   an independent bet. Correcting to `sqrt(periods_per_year / horizon)` cut everything by
   roughly `sqrt(24)` ≈ 4.9×.
2. **The bootstrap was iid.** Resampling overlapping returns independently produced
   intervals several times too narrow. Under the iid bootstrap, the *purged* protocol
   reported a Sharpe of 3.0 with a 95% interval of [1.4, 4.7] — an interval excluding
   zero, on data with no signal in it. Switched to a circular block bootstrap with block
   length equal to the horizon.

Worth stating plainly: **the correct splitter still produced a spurious "significant"
result until the uncertainty estimate was also fixed.** Purging is one of at least three
things that have to be right. That finding is now the reason row 4 exists in the results
table.

**Third bug, found by inspection.** The lookahead-feature leak did nothing — the
contaminated protocol scored the same as the clean one. Cause: the future return was
mixed into noise without standardising first, so a signal of scale ~0.01 was drowned by
noise of scale 1. A leak injector that silently fails to leak is the worst possible bug
in this project, since it would have produced the reassuring conclusion that lookahead
features don't matter much.

After the fix, the contaminated-feature protocol reports 1.28 [0.64, 1.88] with a
perfectly clean splitter. That is the most useful single number in the repo.

---

## 2026-09-25 — agent evaluation harness

Deterministic mock provider first, real providers second. Deliberate: the scoring code
and failure taxonomy needed testing against known ground truth before any number from a
real model could be trusted.

**Scoring bug caught by the failure breakdown, not by the accuracy number.** Aggregate
accuracy looked plausible at ~0.73, but `missing_field` was the dominant failure mode for
every provider at 44–51%, which was too uniform to be real. Cause: about a fifth of
generated listings genuinely lack a field, with ground truth `null`, and omitting the key
was being scored as a miss. Models were being penalised for the correct behaviour.

This is the argument for the taxonomy in one example. The aggregate number was
unremarkable and would not have prompted a second look. The *distribution* of failures
was obviously wrong.

---

## 2026-09-25 — transformer from primitives

Twelve correctness tests before any benchmarking, because a KV cache bug doesn't raise —
it makes generation faster and subtly wrong.

**The optimisation table mostly failed, and that's the useful result.** bf16 autocast is a
2.9× regression on 2 CPU threads with no AMX or AVX512-BF16: autocast still inserts every
cast and then runs the arithmetic in emulation. `torch.compile` recovered 0.34× → 0.35×,
which at this graph size is noise.

**Memory reporting was broken and looked fine.** The first run printed an identical
889 MB for four different configurations. Cause: max RSS is monotonic over the process
lifetime, so every row reported the high-water mark of all previous rows. Now returns NaN
on CPU with a comment, because a number known to be meaningless is worse than no number.

**Generalisation gap climbing from −0.007 to +0.075 over 1,200 steps** and still rising at
the end. The run should probably stop around step 500–600. Worth confirming with a longer
run before believing the eyeball.

---

## Next

- 01: multi-asset panel to exercise the survivorship leak, which is the one
  cross-validation cannot fix because the contamination is in the sampling frame.
- 02: retry policy as an evaluated variable rather than an assumption; an LLM judge as a
  *second* grader scored against the deterministic one.
- 03: rerun the optimisation table on a GPU; the interesting question is the crossover
  model size where bf16 starts winning.
