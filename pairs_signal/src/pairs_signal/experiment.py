"""Run a model over a fixed set and score it: accuracy, cost and latency per decision.

Models:
  teacher   Claude (default claude-haiku-5-5) via ANTHROPIC_API_KEY, JSON output with a reason
  student   the distilled model behind Runpod / vLLM (see student.py)
  rules     offline keyword baseline, no network

With Langfuse keys set (and not --local), items come from the Langfuse dataset
pairs-<task>-<split> (upload with `langfuse_datasets`), the run is a Langfuse dataset run,
every call is a traced generation and the scores accuracy (0/1) / cost_usd / latency_s land
on the traces, with accuracy and averages as run-level scores. Every run is also written locally to
data/out/runs/<run>.jsonl, which export_errors and backtest read.

--dry-run forces the rules model on the local set: no keys, no network.
For the cases task the summary adds the trade result of following the model (trade only
when it says "reverts").
"""

import argparse
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from pairs_signal import config, tasks
from pairs_signal.label_headlines import rule_label
from pairs_signal.llm import langfuse_enabled

RUNS = config.OUT / "runs"


def teacher_predictor(task: tasks.Task, model: str):
    from pairs_signal.llm import Claude

    claude = Claude(model, effort="low", max_tokens=2048)
    labels = list(task.labels)
    if task.name == "cases":
        system = task.system_prompt.replace("Answer with the label only.", "Answer in JSON with the decision and a one-sentence reason.")
        schema = {
            "type": "object",
            "properties": {"decision": {"type": "string", "enum": labels}, "reason": {"type": "string"}},
            "required": ["decision", "reason"],
            "additionalProperties": False,
        }
    else:
        system = task.system_prompt.replace("Answer with the label only.", "Answer in JSON.")
        schema = {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": labels}},
            "required": ["label"],
            "additionalProperties": False,
        }

    def predict(text: str, meta: dict) -> dict:
        r = claude.json_call(system, text, schema, name=f"{task.name}-teacher", metadata=meta)
        label = r.parsed.get("decision") or r.parsed.get("label")
        return {"label": label, "raw": r.parsed, "cost_usd": r.cost_usd, "latency_s": r.latency_s}

    predict.flush = claude.flush
    return predict


def student_predictor(task: tasks.Task, client_dir: str | None):
    from pairs_signal.student import Student

    student = Student(client_dir=client_dir)

    def predict(text: str, meta: dict) -> dict:
        raw, latency, cost = student.complete(task.system_prompt, text, name=f"{task.name}-student", metadata=meta)
        return {"label": task.parse(raw), "raw": raw, "cost_usd": cost, "latency_s": latency}

    predict.flush = lambda: student.langfuse.flush() if student.langfuse else None
    return predict


def rules_predictor(task: tasks.Task):
    shock = tasks.shock_labels()

    def predict(text: str, meta: dict) -> dict:
        start = time.perf_counter()
        if task.name == "headlines":
            label = rule_label(text)
        else:
            m = re.search(r'Latest headline \([^)]*\): "(.*)"', text)
            label = "no_revert" if m and rule_label(m.group(1)) in shock else "reverts"
        return {"label": label, "raw": label, "cost_usd": 0.0, "latency_s": time.perf_counter() - start}

    predict.flush = lambda: None
    return predict


def summarize(task: tasks.Task, rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(r["predicted"] == r["expected"] for r in rows)
    per_label = {}
    for lab in task.labels:
        sub = [r for r in rows if r["expected"] == lab]
        if sub:
            per_label[lab] = {"n": len(sub), "accuracy": round(sum(r["predicted"] == lab for r in sub) / len(sub), 3)}
    cost = sum(r["cost_usd"] for r in rows)
    summary = {
        "n": n,
        "accuracy": round(correct / n, 4) if n else None,
        "unparsed": sum(r["predicted"] is None for r in rows),
        "per_label": per_label,
        "cost_usd_total": round(cost, 6),
        "cost_usd_per_decision": round(cost / n, 8) if n else None,
        "latency_s_mean": round(sum(r["latency_s"] for r in rows) / n, 3) if n else None,
        "confusions": Counter(f"{r['expected']}->{r['predicted']}" for r in rows if r["predicted"] != r["expected"]).most_common(10),
    }
    if task.name == "cases":
        trades = [r for r in rows if r["predicted"] == "reverts"]
        summary["trading"] = {
            "trades": len(trades),
            "hit_rate": round(sum(r["expected"] == "reverts" for r in trades) / len(trades), 4) if trades else None,
            "mean_net_return": round(sum(r["meta"]["net_return"] for r in trades) / len(trades), 6) if trades else None,
            "sum_net_return": round(sum(r["meta"]["net_return"] for r in trades), 6),
            "base_rate_reverts": round(sum(r["expected"] == "reverts" for r in rows) / n, 4) if n else None,
        }
    return summary


def item_evaluations(output: dict, expected: str, meta: dict, run_name: str):
    from langfuse import Evaluation

    ok = output["label"] == expected
    info = {"id": meta.get("id"), "expected": expected, "predicted": output["label"], "run": run_name}
    return [
        Evaluation(name="accuracy", value=1.0 if ok else 0.0, comment=f"expected {expected}, got {output['label']}", metadata=info),
        Evaluation(name="cost_usd", value=float(output["cost_usd"])),
        Evaluation(name="latency_s", value=float(output["latency_s"])),
    ]


def run_langfuse(task, predict, split: str, run_name: str, workers: int, model_label: str) -> list[dict]:
    from langfuse import Evaluation, get_client

    lf = get_client()
    dataset = lf.get_dataset(f"pairs-{task.name}-{split}")

    def task_fn(*, item, **kwargs):
        meta = dict(item.metadata or {})
        return predict(item.input["text"], {"id": meta.get("id"), "split": split})

    def evaluator(*, input, output, expected_output, metadata, **kwargs):
        return item_evaluations(output, expected_output["label"], metadata or {}, run_name)

    def run_evaluator(*, item_results, **kwargs):
        outs = [r for r in item_results if r.output]
        n = max(len(outs), 1)
        acc = sum(r.output["label"] == r.item.expected_output["label"] for r in outs) / n
        return [
            Evaluation(name="accuracy_run", value=acc),
            Evaluation(name="cost_usd_per_decision", value=sum(r.output["cost_usd"] for r in outs) / n),
            Evaluation(name="latency_s_mean", value=sum(r.output["latency_s"] for r in outs) / n),
        ]

    result = dataset.run_experiment(
        name=f"{task.name}-{model_label}",
        run_name=run_name,
        task=task_fn,
        evaluators=[evaluator],
        run_evaluators=[run_evaluator],
        max_concurrency=workers,
        metadata={"model": model_label, "split": split},
    )
    lf.flush()
    print(result.format())
    rows = []
    for r in result.item_results:
        if not r.output:
            continue
        meta = dict(r.item.metadata or {})
        rows.append({"id": meta.get("id"), "input": r.item.input["text"], "expected": r.item.expected_output["label"], "predicted": r.output["label"],
                     "raw": r.output["raw"], "cost_usd": r.output["cost_usd"], "latency_s": r.output["latency_s"], "trace_id": r.trace_id, "meta": meta})
    return rows


def run_local(task, predict, split: str, workers: int) -> list[dict]:
    examples = [json.loads(line) for line in (config.OUT / "sets" / f"{task.name}_{split}.jsonl").open()]

    def one(ex: dict) -> dict:
        out = predict(ex["input"], {"id": ex["id"], "split": split})
        return {"id": ex["id"], "input": ex["input"], "expected": ex["label"], "predicted": out["label"], "raw": out["raw"],
                "cost_usd": out["cost_usd"], "latency_s": out["latency_s"], "trace_id": None, "meta": {"id": ex["id"], **ex["meta"]}}

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, examples))


def main(default_task: str = "headlines", default_model: str = "teacher") -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["headlines", "cases"], default=default_task)
    parser.add_argument("--model", choices=["teacher", "student", "rules"], default=default_model)
    parser.add_argument("--split", choices=["train", "val", "test", "case_test"], default="test", help="case_test: headlines of test-period cases (backtest c)")
    parser.add_argument("--teacher-model", default=config.TEACHER_MODEL)
    parser.add_argument("--client-dir", default=None, help="dir with distil labs model_client.py (student)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--local", action="store_true", help="read the local set even if Langfuse is configured")
    parser.add_argument("--dry-run", action="store_true", help="rules model on the local set, no network")
    args = parser.parse_args()

    task = tasks.get(args.task)
    model = "rules" if args.dry_run else args.model
    model_label = {"teacher": args.teacher_model, "student": "student", "rules": "rules"}[model]
    run_name = args.run_name or f"{args.task}-{args.split}-{model_label}-{time.strftime('%Y%m%d-%H%M%S')}"
    if model == "teacher":
        predict = teacher_predictor(task, args.teacher_model)
    elif model == "student":
        predict = student_predictor(task, args.client_dir)
    else:
        predict = rules_predictor(task)

    use_langfuse = langfuse_enabled() and not (args.local or args.dry_run)
    print(f"run {run_name}: {model_label} on {args.task}/{args.split} ({'Langfuse dataset' if use_langfuse else 'local set'})")
    if use_langfuse:
        rows = run_langfuse(task, predict, args.split, run_name, args.workers, model_label)
    else:
        rows = run_local(task, predict, args.split, args.workers)
    predict.flush()

    summary = {"run": run_name, "task": args.task, "split": args.split, "model": model_label, **summarize(task, rows)}
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / f"{run_name}.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    (RUNS / f"{run_name}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_label"}, indent=2))
    print(f"-> {RUNS / run_name}.jsonl")


def baseline() -> None:
    """Spec entry point: the large model on the fixed case test set."""
    main(default_task="cases", default_model="teacher")


if __name__ == "__main__":
    main()
