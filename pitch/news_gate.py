"""Live demo: one stock headline in, event type (or sentiment) out.

Run from pairs_signal/ so the pairs_signal package and its uv environment are used:
  uv run python ../pitch/news_gate.py "Coca-Cola names John Murphy CFO"
  uv run python ../pitch/news_gate.py --model teacher "Coca-Cola names John Murphy CFO"
  uv run python ../pitch/news_gate.py --task news_sentiment "Coca-Cola names John Murphy CFO"
  uv run python ../pitch/news_gate.py --dry-run "Coca-Cola names John Murphy CFO"

student  distilled model behind RUNPOD_ENDPOINT_ID or STUDENT_BASE_URL (see student.py)
teacher  Claude Haiku 5.5, for the side-by-side cost and latency comparison
--dry-run keyword rules, no keys and no network, for rehearsal only

With LANGFUSE_* keys set every call lands in Langfuse as a generation named news-gate-live.
"""

import argparse
import json

from pairs_signal import config, tasks
from pairs_signal.experiment import rules_predictor


def shock_labels(task_name: str) -> frozenset[str]:
    """Event labels that mean the company itself changed; empty for tasks without that rule."""
    path = config.TASKS / ("headline_labels.json" if task_name == "headlines" else f"{task_name}_labels.json")
    return frozenset(json.loads(path.read_text()).get("fundamental_shock_labels", []))


def classify(task: tasks.Task, model: str, headline: str, client_dir: str | None) -> tuple[str | None, float, float]:
    """Returns (label, latency seconds, cost USD) and flushes Langfuse."""
    if model == "rules":
        r = rules_predictor(task)(headline, {})
        return r["label"], r["latency_s"], 0.0
    metadata = {"model": model, "task": task.name}
    if model == "teacher":
        from pairs_signal.llm import Claude

        claude = Claude(config.TEACHER_MODEL, effort="low", max_tokens=2048)
        schema = {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": list(task.labels)}},
            "required": ["label"],
            "additionalProperties": False,
        }
        system = task.system_prompt.replace("Answer with the label only.", "Answer in JSON.")
        r = claude.json_call(system, headline, schema, name="news-gate-live", metadata=metadata)
        claude.flush()
        return r.parsed["label"], r.latency_s, r.cost_usd
    from pairs_signal.student import Student

    student = Student(client_dir=client_dir)
    raw, latency, cost = student.complete(task.system_prompt, headline, name="news-gate-live", metadata=metadata)
    if student.langfuse:
        student.langfuse.flush()
    return task.parse(raw), latency, cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("headline")
    parser.add_argument("--task", choices=["news_event", "news_sentiment", "headlines"], default="news_event")
    parser.add_argument("--model", choices=["student", "teacher", "rules"], default="student")
    parser.add_argument("--client-dir", default=None, help="dir with distil labs model_client.py (student)")
    parser.add_argument("--dry-run", action="store_true", help="keyword rules, no network")
    args = parser.parse_args()
    model = "rules" if args.dry_run else args.model
    task = tasks.get(args.task)

    label, latency, cost = classify(task, model, args.headline, args.client_dir)
    print(f"headline  {args.headline}")
    print(f"model     {model}")
    print(f"label     {label or 'unparsed'}")
    shocks = shock_labels(args.task)
    if shocks:
        if label is None:
            print("shock     unknown (answer not parsed)")
        else:
            print(f"shock     {'yes, fundamental shock' if label in shocks else 'no'}")
    print(f"latency   {latency * 1000:.0f} ms")
    print(f"cost      ${cost:.6f}  (${cost * 1000:.4f} per 1,000 calls)")


if __name__ == "__main__":
    main()
