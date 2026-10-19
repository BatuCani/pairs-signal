from abc import ABC, abstractmethod
import json
from pathlib import Path
from time import perf_counter

from pair_agent.tracing.schemas import Trace, TraceStep


class TraceSink(ABC):
    @property
    @abstractmethod
    def tool_call_count(self) -> int: ...
    @property
    @abstractmethod
    def model_call_count(self) -> int: ...
    @abstractmethod
    def start_trace(self, **metadata) -> str: ...
    @abstractmethod
    def log_tool_call(self, name: str, inputs: dict, output: dict, latency_ms: float = 0,
                      error: str | None = None): ...
    @abstractmethod
    def log_model_call(self, inputs: dict, output: dict, latency_ms: float = 0,
                       error: str | None = None): ...
    @abstractmethod
    def log_score(self, scores: dict): ...
    @abstractmethod
    def end_trace(self, result: dict, status: str = "completed") -> Path: ...


class LocalJsonlTraceSink(TraceSink):
    """One completed JSON object per file; one sink per sequential agent instance."""
    def __init__(self, directory: Path = Path("data/traces")):
        self.directory = Path(directory)
        self.trace: Trace | None = None

    @property
    def tool_call_count(self):
        return self.trace.tool_call_count if self.trace else 0

    @property
    def model_call_count(self):
        return sum(s.type == "model_call" for s in self.trace.steps) if self.trace else 0

    def start_trace(self, **metadata):
        self.trace = Trace(**metadata)
        self.started = perf_counter()
        return self.trace.trace_id

    def log_tool_call(self, name, inputs, output, latency_ms=0, error=None):
        self.trace.steps.append(TraceStep(type="tool_call", name=name, input=inputs,
                                          output=output, latency_ms=latency_ms, error=error))
        self.trace.tool_call_count += 1

    def log_model_call(self, inputs, output, latency_ms=0, error=None):
        self.trace.steps.append(TraceStep(type="model_call", name=self.trace.model or "model",
                                          input=inputs, output=output, latency_ms=latency_ms, error=error))

    def log_score(self, scores):
        self.trace.scores.update(scores)

    def end_trace(self, result, status="completed"):
        self.trace.final_answer = result
        selected = result.get("selected_pairs", [])
        self.trace.selected_pair = [selected[0]["ticker_a"], selected[0]["ticker_b"]] if selected else None
        self.trace.failure_modes = result.get("failure_modes", [])
        self.trace.status = status
        self.trace.latency_ms = (perf_counter() - self.started) * 1000
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self.trace.trace_id}.jsonl"
        temp = path.with_suffix(".tmp")
        # Pydantic validates finite values; strict JSON forbids NaN/Infinity.
        temp.write_text(json.dumps(self.trace.model_dump(mode="json"), allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(path)
        return path
