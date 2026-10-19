import json
from pathlib import Path

from pair_agent.models.schemas import ResearchRequest
from pair_agent.tracing.schemas import Trace


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if "id" not in item:
                raise ValueError("task requires id")
            ResearchRequest.model_validate({k: v for k, v in item.items() if k != "id"})
            tasks.append(item)
    if len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("duplicate task ids")
    return tasks


def select_traces(directory: Path, min_score: float = 0, max_score: float = 1,
                  failures_only: bool = False) -> list[Trace]:
    if not 0 <= min_score <= max_score <= 1:
        raise ValueError("require 0 <= min_score <= max_score <= 1")
    traces = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            trace = Trace.model_validate_json(line)
            score = trace.scores.get("overall_score")
            if trace.status != "completed" or score is None:
                continue
            if min_score <= score <= max_score and (not failures_only or trace.failure_modes):
                traces.append(trace)
    return traces


def export_training_dataset(directory: Path = Path("data/traces"),
                             output: Path = Path("data/processed/training.jsonl"),
                             min_score: float = .8, max_score: float = 1,
                             failures_only: bool = False) -> int:
    traces = select_traces(directory, min_score, max_score, failures_only)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for trace in traces:
            rows = [dict(role="user", content=trace.request["request"])]
            # Preserve actual provider conversations when available, including tool IDs.
            calls = [s for s in trace.steps if s.type == "model_call" and s.output.get("message")]
            if calls:
                last_call = calls[-1]
                message = last_call.output["message"]
                rows = list(last_call.input.get("messages", rows)) + [message]
                remaining = trace.steps[trace.steps.index(last_call) + 1:]
                for call in message.get("tool_calls") or []:
                    name = call["function"]["name"]
                    matching = [s for s in remaining if s.type == "tool_call" and s.name == name]
                    output = matching[-1].output if matching else {"error": "no completed tool response"}
                    rows.append(dict(role="tool", tool_call_id=call["id"], content=json.dumps(output)))
            # Final answer is always reconstructed from deterministic result objects.
            rows.append(dict(role="assistant", content=json.dumps(trace.final_answer, ensure_ascii=False)))
            metadata = dict(trace_id=trace.trace_id, overall_score=trace.scores["overall_score"],
                            failure_modes=trace.failure_modes, source=trace.source,
                            agent_version=trace.agent_version, model=trace.model,
                            config=trace.config, steps=[s.model_dump(mode="json") for s in trace.steps])
            handle.write(json.dumps(dict(messages=rows, metadata=metadata), allow_nan=False) + "\n")
    return len(traces)


def export_failure_dataset(directory: Path = Path("data/traces"),
                           output: Path = Path("data/processed/failures.jsonl"),
                           max_score: float = .5) -> int:
    return export_training_dataset(directory, output, 0, max_score, failures_only=True)
