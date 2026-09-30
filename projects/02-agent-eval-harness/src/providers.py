"""Model backends behind one interface, plus a deterministic mock.

The mock exists so the harness is runnable by anyone, immediately, with no API key and
no network. A reviewer who has to provision credentials before seeing anything work
mostly does not see it work.

Real providers are thin on purpose. This project is about the evaluation, not about
wrapping SDKs.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Completion:
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens. Update from the provider's page before quoting costs."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_mtok + completion_tokens * self.output_per_mtok
        ) / 1_000_000


class Provider(ABC):
    name: str
    pricing: Pricing

    @abstractmethod
    def complete(self, prompt: str, *, max_tokens: int = 512) -> Completion: ...

    def _timed(self, fn):
        start = time.perf_counter()
        try:
            text, pt, ct, err = fn()
        except Exception as exc:  # noqa: BLE001 - provider errors are data, not crashes
            return Completion("", (time.perf_counter() - start) * 1000, 0, 0, str(exc))
        return Completion(text, (time.perf_counter() - start) * 1000, pt, ct, err)


class MockProvider(Provider):
    """Deterministic stand-in with configurable, realistic failure modes.

    Every failure rate below corresponds to something observed in real runs. Keeping them
    tunable means the scoring and taxonomy code can be tested against known ground truth
    — which is the same argument as the synthetic series in project 01.
    """

    def __init__(
        self,
        name: str = "mock-baseline",
        *,
        seed: int = 0,
        accuracy: float = 0.72,
        json_failure_rate: float = 0.06,
        omission_rate: float = 0.08,
        hallucination_rate: float = 0.05,
        base_latency_ms: float = 240.0,
    ) -> None:
        self.name = name
        self.pricing = Pricing(0.0, 0.0)
        self.seed = seed
        self.accuracy = accuracy
        self.json_failure_rate = json_failure_rate
        self.omission_rate = omission_rate
        self.hallucination_rate = hallucination_rate
        self.base_latency_ms = base_latency_ms

    def complete(self, prompt: str, *, max_tokens: int = 512) -> Completion:
        rng = random.Random(f"{self.seed}:{self.name}:{prompt}")
        truth = self._recover_truth(prompt)
        latency = self.base_latency_ms * (0.6 + rng.random() * 1.6)

        if rng.random() < self.json_failure_rate:
            return Completion(
                "Sure! Here's the extraction:\n" + json.dumps(truth)[:-4],
                latency, 420, 90,
            )

        out: dict[str, object] = {}
        for field, value in truth.items():
            r = rng.random()
            if r < self.omission_rate:
                continue
            if r < self.omission_rate + self.hallucination_rate:
                out[field] = "unknown" if rng.random() < 0.5 else f"{value}X"
                continue
            out[field] = value if rng.random() < self.accuracy else self._corrupt(value, rng)
        return Completion(json.dumps(out), latency, 420, 90)

    @staticmethod
    def _corrupt(value: object, rng: random.Random) -> object:
        s = str(value)
        if not s:
            return s
        if rng.random() < 0.5:
            return s.upper() if s.islower() else s.lower()
        return s[: max(1, len(s) - 2)]

    @staticmethod
    def _recover_truth(prompt: str) -> dict[str, str]:
        """The mock reads the answer key embedded in the prompt by the harness.

        This is deliberate and it is flagged loudly: the mock is a *plumbing* test, not a
        capability test. Its numbers are meaningless as model comparisons and the report
        marks them as such.
        """
        m = re.search(r"<!--truth:(.*?)-->", prompt, flags=re.S)
        return json.loads(m.group(1)) if m else {}


class OpenAICompatibleProvider(Provider):
    """Works against OpenAI, Groq, Together, or a local Ollama /v1 endpoint."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        pricing: Pricing | None = None,
        name: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.name = name or model
        self.pricing = pricing or Pricing(0.0, 0.0)

    def complete(self, prompt: str, *, max_tokens: int = 512) -> Completion:
        import urllib.request

        def _call():
            body = json.dumps(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                }
            ).encode()
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read())
            usage = payload.get("usage", {})
            return (
                payload["choices"][0]["message"]["content"],
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
                None,
            )

        return self._timed(_call)


def default_providers() -> list[Provider]:
    """Three mocks with different characteristics — enough to exercise the report.

    Swap in real providers by editing this function. The harness does not care.
    """
    return [
        MockProvider("mock-small", seed=1, accuracy=0.61, json_failure_rate=0.11,
                     base_latency_ms=120),
        MockProvider("mock-baseline", seed=2, accuracy=0.72, json_failure_rate=0.06,
                     base_latency_ms=240),
        MockProvider("mock-large", seed=3, accuracy=0.86, json_failure_rate=0.02,
                     base_latency_ms=900),
    ]
