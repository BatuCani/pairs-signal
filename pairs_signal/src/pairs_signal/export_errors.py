"""Improvement loop: turn a run's wrong predictions into a v2 distil labs dataset.

Errors come from Langfuse (item scores named "accuracy" with value 0 whose metadata.run is the
run name, written by `experiment`) or, with --local or without Langfuse keys, from the
local run file data/out/runs/<run>.jsonl.

Only train/val runs are accepted: the fixed test set never feeds training.

Writes data/out/distil/<task>_v2/: the v1 bundle plus
  - every misclassified example added to train.jsonl with its gold label
  - the label mutator reweighted toward the labels the model got wrong
  - data/out/runs/<run>.errors.csv, errors sorted by confusion (expected -> predicted)
"""

import argparse
import csv
import json
import re
import shutil
from collections import Counter

from pairs_signal import config, tasks
from pairs_signal.llm import langfuse_enabled

RUNS = config.OUT / "runs"


def errors_from_langfuse(run_name: str) -> list[dict]:
    from langfuse import get_client

    lf = get_client()
    out, cursor = [], None
    while True:
        page = lf.api.scores.get_many(name="accuracy", value_max=0.5, limit=100, cursor=cursor)
        for s in page.data:
            meta = s.metadata or {}
            if meta.get("run") == run_name:
                out.append({"id": meta.get("id"), "expected": meta.get("expected"), "predicted": meta.get("predicted")})
        cursor = getattr(page.meta, "next_cursor", None) or getattr(page.meta, "cursor", None)
        if not cursor or not page.data:
            return out


def errors_from_file(run_name: str) -> list[dict]:
    rows = [json.loads(line) for line in (RUNS / f"{run_name}.jsonl").open()]
    return [{"id": r["id"], "expected": r["expected"], "predicted": r["predicted"]} for r in rows if r["predicted"] != r["expected"]]


def reweighted_config(text: str, weights: dict[str, int], labels: list[str]) -> str:
    """Replace the adaptive label mutator by fixed weights favouring error-heavy labels."""
    dist = ", ".join(str(1 + weights.get(lab, 0)) for lab in labels)
    text = text.replace("      method: adaptive\n", f"      # v2: weights = 1 + validation errors per label, in value order\n      target_distribution: [{dist}]\n")
    return re.sub(r"(base:\n)", r"\1  # v2 of the dataset: v1 + misclassified validation examples\n", text, count=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_name")
    parser.add_argument("--local", action="store_true", help="read errors from the local run file")
    parser.add_argument("--version", default="v2")
    args = parser.parse_args()

    summary = json.loads((RUNS / f"{args.run_name}.summary.json").read_text())
    if summary["split"] in ("test", "case_test", "phrasebank_test"):
        raise SystemExit("refusing to train on the fixed test set: run the experiment with --split val")
    task = tasks.get(summary["task"])
    use_langfuse = langfuse_enabled() and not args.local
    errors = errors_from_langfuse(args.run_name) if use_langfuse else errors_from_file(args.run_name)
    print(f"{len(errors)} errors in {args.run_name} (from {'Langfuse' if use_langfuse else 'local run file'})")

    examples = {e["id"]: e for e in tasks.load_examples(task.name)}
    test_inputs = {json.loads(line)["input"] for line in (config.OUT / "sets" / f"{task.name}_test.jsonl").open()}
    errors = [e for e in errors if e["id"] in examples and examples[e["id"]]["input"] not in test_inputs]
    errors.sort(key=lambda e: (e["expected"], str(e["predicted"])))

    with (RUNS / f"{args.run_name}.errors.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "expected", "predicted", "input"])
        for e in errors:
            writer.writerow([e["id"], e["expected"], e["predicted"], examples[e["id"]]["input"]])
    for pair, n in Counter(f"{e['expected']} -> {e['predicted']}" for e in errors).most_common():
        print(f"  {n:4d}  {pair}")

    src = config.OUT / "distil" / task.name
    dst = config.OUT / "distil" / f"{task.name}_{args.version}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    train_inputs = {json.loads(line)["messages"][0]["content"] for line in (dst / "train.jsonl").open()}
    added = 0
    with (dst / "train.jsonl").open("a") as fh:
        for e in errors:
            ex = examples[e["id"]]
            if ex["input"] in train_inputs:
                continue
            fh.write(json.dumps({"messages": [{"role": "user", "content": ex["input"]}, {"role": "assistant", "content": ex["label"]}]}) + "\n")
            train_inputs.add(ex["input"])
            added += 1
    weights = Counter(e["expected"] for e in errors)
    (dst / "config.yaml").write_text(reweighted_config((src / "config.yaml").read_text(), weights, list(task.labels)))
    print(f"{added} examples added -> {dst}")


if __name__ == "__main__":
    main()
