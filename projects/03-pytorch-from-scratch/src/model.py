"""A decoder-only transformer written from primitives.

No `nn.Transformer`, no `nn.MultiheadAttention`, no HuggingFace. Attention is written
out as matrix operations so the mechanism is visible and testable.

Two attention implementations are kept side by side:

  `manual`  explicit QK^T / sqrt(d), masked softmax, weighted sum of V
  `sdpa`    torch.nn.functional.scaled_dot_product_attention

They are asserted numerically equivalent in `tests/test_model.py`. Keeping both is the
point of the project: `manual` shows the mechanism, `sdpa` is what you ship, and the
benchmark measures exactly what the swap buys. An optimisation you cannot measure is a
belief.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 128
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False
    attn_impl: str = "manual"  # "manual" | "sdpa"

    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        return self.n_embd // self.n_head


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Shapes, written out because this is where the bugs live:

        x            (B, T, C)
        qkv          (B, T, 3C)      one fused projection, then split
        q, k, v      (B, nh, T, hd)  after view + transpose
        att          (B, nh, T, T)   the quadratic term
        out          (B, T, C)       after transpose + contiguous + view
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim

        # One projection for Q, K and V rather than three. Fewer, larger GEMMs is the
        # single most consequential layout decision in the block.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Registered as a buffer so it moves with .to(device) and is not a parameter.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(
        self, x: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        T_k = k.size(2)

        if self.cfg.attn_impl == "sdpa":
            # During cached decode T == 1 and every cached position is visible, so the
            # causal flag must be off or the single query is masked against itself.
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.cfg.dropout if self.training else 0.0,
                is_causal=(kv_cache is None and T > 1),
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Rows are the T queries ending at position T_k; columns are all T_k keys.
            mask = self.causal_mask[:, :, T_k - T : T_k, :T_k]
            att = att.masked_fill(~mask, float("-inf"))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out)), new_cache


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    """Pre-norm residual block.

    Pre-norm rather than post-norm: it is what makes deep stacks trainable without a
    learning-rate warmup schedule tuned by hand, which matters when you have one machine
    and no patience for babysitting a run.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying. Saves vocab_size * n_embd parameters and reliably improves
        # small-model loss; the input and output embeddings are solving the same problem.
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scaled init on residual projections (GPT-2). Without it the residual stream
        # variance grows with depth and early training is unstable.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def n_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list | None = None,
    ):
        B, T = idx.shape
        offset = kv_caches[0][0].size(2) if kv_caches else 0
        if offset + T > self.cfg.block_size:
            raise ValueError(f"sequence length {offset + T} exceeds block_size")

        pos = torch.arange(offset, offset + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        new_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, kv_caches[i] if kv_caches else None)
            new_caches.append(cache)

        x = self.ln_f(x)

        if targets is None:
            # Only the last position is needed at inference. Computing logits for every
            # position costs a (T x vocab) matmul per step and buys nothing.
            logits = self.head(x[:, [-1], :])
            return logits, None, new_caches

        logits = self.head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
        )
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive sampling, with and without a KV cache.

        Without the cache, every step re-attends over the whole prefix: O(T^2) work to
        emit T tokens. With it, each step attends once over the cached keys. The output
        distribution is identical either way, which `test_kv_cache_matches_full_forward`
        asserts — a cache that changes the output is a bug that looks like a speedup.
        """
        self.eval()
        caches = None

        for _ in range(max_new_tokens):
            if use_cache:
                step_input = idx if caches is None else idx[:, -1:]
            else:
                step_input = idx[:, -self.cfg.block_size :]

            logits, _, new_caches = self(step_input, kv_caches=caches)
            if use_cache:
                caches = new_caches
                if caches[0][0].size(2) >= self.cfg.block_size:
                    caches = None
                    idx = idx[:, -self.cfg.block_size + 1 :]

            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)

        return idx
