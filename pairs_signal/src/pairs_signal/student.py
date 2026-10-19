"""Client for the distilled student served by vLLM (Runpod serverless, a Runpod pod, or local).

Endpoint resolution, first match wins:
  STUDENT_BASE_URL                       any OpenAI-compatible base URL (pod, local vLLM)
  RUNPOD_ENDPOINT_ID (+ RUNPOD_API_KEY)  https://api.runpod.ai/v2/<id>/openai/v1
The served adapter name defaults to "model" (STUDENT_MODEL), as in distil labs' vLLM command.

If --client-dir points at the model_client.py that `distil slm download-metadata` writes, its
DistilLabsLLM is used so sampling and thinking settings match training. Otherwise a plain
chat completion is sent with thinking off and temperature 0.

Every call is a Langfuse generation when LANGFUSE_* keys are set.
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

from pairs_signal.llm import langfuse_enabled

# Serverless GPU price used for cost per decision: billed per worker-second while a request
# runs. Default is an assumption (24 GB flex worker); set RUNPOD_PRICE_PER_SEC to your tier.
DEFAULT_PRICE_PER_SEC = 0.00019


def resolve_base_url() -> str | None:
    if os.environ.get("STUDENT_BASE_URL"):
        return os.environ["STUDENT_BASE_URL"]
    if os.environ.get("RUNPOD_ENDPOINT_ID"):
        return f"https://api.runpod.ai/v2/{os.environ['RUNPOD_ENDPOINT_ID']}/openai/v1"
    return None


class Student:
    def __init__(self, base_url: str | None = None, model: str | None = None, client_dir: str | None = None):
        self.base_url = base_url or resolve_base_url()
        if not self.base_url:
            raise SystemExit("no student endpoint: set STUDENT_BASE_URL or RUNPOD_ENDPOINT_ID")
        self.model = model or os.environ.get("STUDENT_MODEL", "model")
        api_key = os.environ.get("STUDENT_API_KEY") or os.environ.get("RUNPOD_API_KEY") or "EMPTY"
        self.price_per_sec = float(os.environ.get("RUNPOD_PRICE_PER_SEC", DEFAULT_PRICE_PER_SEC))
        self.distil_client = None
        if client_dir:
            path = Path(client_dir) / "model_client.py"
            spec = importlib.util.spec_from_file_location("distil_model_client", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["distil_model_client"] = module
            spec.loader.exec_module(module)
            self.distil_client = module.DistilLabsLLM(model_name=self.model, base_url=self.base_url, api_key=api_key)
        else:
            from openai import OpenAI

            self.client = OpenAI(base_url=self.base_url, api_key=api_key)
        self.langfuse = None
        if langfuse_enabled():
            from langfuse import get_client

            self.langfuse = get_client()

    def _invoke(self, messages: list[dict]) -> str:
        if self.distil_client is not None:
            out = self.distil_client.invoke(messages)
            return out if isinstance(out, str) else (out.get("content") if isinstance(out, dict) else out.content) or ""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content or ""

    def complete(self, system: str, user: str, name: str = "student", metadata: dict | None = None) -> tuple[str, float, float]:
        """Returns (answer text, latency seconds, estimated cost USD)."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.langfuse is None:
            return self._timed(messages)
        with self.langfuse.start_as_current_observation(
            name=name, as_type="generation", model=f"student:{self.model}", input=messages, metadata={"base_url": self.base_url, **(metadata or {})}
        ) as gen:
            text, latency, cost = self._timed(messages)
            gen.update(output=text, cost_details={"total": cost}, metadata={"latency_s": round(latency, 3)})
            return text, latency, cost

    def _timed(self, messages: list[dict]) -> tuple[str, float, float]:
        start = time.perf_counter()
        text = self._invoke(messages)
        latency = time.perf_counter() - start
        return text, latency, latency * self.price_per_sec
