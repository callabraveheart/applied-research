"""Correctness tests for attention, causal masking, and the KV cache.

The KV cache test is the one that earns its place. A cache bug does not raise — it makes
generation faster and subtly wrong, which is indistinguishable from a successful
optimisation unless you assert equivalence. Same argument as project 01: the dangerous
failures are the flattering ones.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.model import GPT, CausalSelfAttention, GPTConfig


def _cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=32, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    return GPTConfig(**(base | kw))


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)


def test_manual_and_sdpa_attention_agree():
    """The two implementations must be numerically identical, or the benchmark is lying."""
    x = torch.randn(2, 8, 32)

    manual = CausalSelfAttention(_cfg(attn_impl="manual")).eval()
    sdpa = CausalSelfAttention(_cfg(attn_impl="sdpa")).eval()
    sdpa.load_state_dict(manual.state_dict())

    with torch.no_grad():
        a, _ = manual(x)
        b, _ = sdpa(x)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_manual_attention_matches_hand_computation():
    """Check the mechanism against the formula, not against another library."""
    cfg = _cfg(n_head=1, n_embd=8, attn_impl="manual")
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(1, 4, 8)

    with torch.no_grad():
        got, _ = attn(x)

        q, k, v = attn.qkv(x).split(8, dim=2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(8)
        mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        expected = attn.proj(F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v)

    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_causality_future_tokens_cannot_change_the_past(impl):
    """The defining property. Perturb the last token; earlier outputs must not move."""
    model = GPT(_cfg(attn_impl=impl)).eval()
    idx = torch.randint(0, 32, (1, 8))

    with torch.no_grad():
        logits_a, _, _ = model(idx, targets=torch.zeros_like(idx))
        altered = idx.clone()
        altered[0, -1] = (altered[0, -1] + 7) % 32
        logits_b, _, _ = model(altered, targets=torch.zeros_like(idx))

    torch.testing.assert_close(logits_a[:, :-1], logits_b[:, :-1], rtol=1e-4, atol=1e-5)
    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1])


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_kv_cache_matches_full_forward(impl):
    """Incremental decode with a cache must equal recomputing the whole prefix.

    This is the test that makes the cache trustworthy. Without it, a cache that drops a
    position or mis-indexes the positional embedding shows up only as slightly worse
    samples, months later.
    """
    model = GPT(_cfg(attn_impl=impl)).eval()
    idx = torch.randint(0, 32, (1, 6))

    with torch.no_grad():
        full_logits, _, _ = model(idx)

        caches = None
        for t in range(idx.size(1)):
            step = idx[:, : t + 1] if caches is None else idx[:, t : t + 1]
            cached_logits, _, caches = model(step, kv_caches=caches)

    torch.testing.assert_close(full_logits, cached_logits, rtol=1e-4, atol=1e-5)


def test_cache_grows_by_one_position_per_step():
    model = GPT(_cfg()).eval()
    idx = torch.randint(0, 32, (1, 3))
    with torch.no_grad():
        _, _, caches = model(idx)
        assert caches[0][0].size(2) == 3
        _, _, caches = model(idx[:, -1:], kv_caches=caches)
        assert caches[0][0].size(2) == 4


def test_generation_respects_requested_length():
    model = GPT(_cfg()).eval()
    idx = torch.randint(0, 32, (1, 2))
    for use_cache in (True, False):
        out = model.generate(idx, max_new_tokens=5, use_cache=use_cache)
        assert out.shape == (1, 7)


def test_weights_are_tied():
    model = GPT(_cfg())
    assert model.head.weight is model.tok_emb.weight


def test_untrained_loss_is_near_the_uniform_floor():
    """A correctly initialised model starts at ln(vocab_size). Anything else is a bug.

    Cheapest possible smoke test for initialisation, and it catches scale errors in the
    output layer immediately.
    """
    cfg = _cfg()
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (4, 8))
    with torch.no_grad():
        _, loss, _ = model(idx, targets=torch.randint(0, cfg.vocab_size, (4, 8)))
    assert abs(float(loss) - math.log(cfg.vocab_size)) < 0.25


def test_sequence_longer_than_block_size_is_rejected():
    cfg = _cfg(block_size=8)
    model = GPT(cfg)
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.randint(0, 32, (1, 9)))


def test_mismatched_head_dimension_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        _ = GPTConfig(n_embd=30, n_head=4).head_dim
