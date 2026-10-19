"""Thin Claude client: JSON-schema output, token cost accounting, optional Langfuse tracing.

Credentials come from the environment only: ANTHROPIC_API_KEY for Claude and, if set,
LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST (or LANGFUSE_BASE_URL) for tracing.
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field

# USD per million tokens (input, output), prompts up to 100K tokens.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def langfuse_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, inp: int, out: int, cost: float) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += inp
            self.output_tokens += out
            self.cost_usd += cost


@dataclass
class CallResult:
    parsed: dict
    cost_usd: float
    latency_s: float
    trace_id: str | None = None


class Claude:
    def __init__(self, model: str, effort: str = "low", max_tokens: int = 4096):
        import anthropic

        if model not in PRICES:
            raise ValueError(f"no price for {model}; add it to PRICES")
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.usage = Usage()
        self.langfuse = None
        if langfuse_enabled():
            from langfuse import Langfuse

            host = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
            self.langfuse = Langfuse(base_url=host) if host else Langfuse()

    def cost(self, inp: int, out: int) -> float:
        price_in, price_out = PRICES[self.model]
        return (inp * price_in + out * price_out) / 1e6

    def json_call(self, system: str, user: str, schema: dict, name: str, metadata: dict | None = None) -> CallResult:
        """One structured call, traced as a Langfuse generation when tracing is on."""
        if self.langfuse is None:
            return self._call(system, user, schema)
        with self.langfuse.start_as_current_observation(
            name=name,
            as_type="generation",
            model=self.model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            metadata=metadata,
            model_parameters={"effort": self.effort, "max_tokens": self.max_tokens},
        ) as gen:
            result = self._call(system, user, schema, gen)
            result.trace_id = self.langfuse.get_current_trace_id()
            return result

    def _call(self, system: str, user: str, schema: dict, gen=None) -> CallResult:
        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        latency = time.perf_counter() - start
        inp, out = response.usage.input_tokens, response.usage.output_tokens
        cost = self.cost(inp, out)
        self.usage.add(inp, out, cost)
        if response.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"stop_reason={response.stop_reason} (request {response._request_id})")
        text = next(b.text for b in response.content if b.type == "text")
        parsed = json.loads(text)
        if gen is not None:
            gen.update(output=parsed, usage_details={"input": inp, "output": out}, cost_details={"total": cost})
        return CallResult(parsed, cost, latency)

    def score(self, trace_id: str | None, name: str, value: float, comment: str | None = None) -> None:
        if self.langfuse is not None and trace_id:
            self.langfuse.create_score(trace_id=trace_id, name=name, value=value, comment=comment)

    def flush(self) -> None:
        if self.langfuse is not None:
            self.langfuse.flush()
