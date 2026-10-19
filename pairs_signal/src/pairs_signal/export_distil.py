"""Write fixed train/val/test sets and distil labs upload directories for both tasks.

distil labs deprecated the classification task type; a fixed label set is a
question-answering task whose job description carries the labels and their definitions,
with one mutator over the labels (docs: build/task-selection, "Classifying into labels").

Outputs per task (headlines, cases):
  data/out/sets/<task>_{train,val,test}.jsonl   {id, input, label, meta}; test is the fixed
                                                test set used everywhere, never for training
  data/out/sets/headlines_case_test.jsonl       headlines of test-period cases (backtest c)
  data/out/distil/<task>/                       config.yaml, job_description.json,
                                                train.jsonl, test.jsonl[, traces.jsonl]
  data/out/distil/news_event_small/             news_event with 10 train rows per label
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from pairs_signal import config, tasks

# Balanced per label, capped per split (headline noise is ~2/3 of the raw stream; the news
# sources are skewed toward macro and company news). cases keep their natural distribution.
CAPS = {
    "news_event": {"train": 150, "val": 30, "test": 40},
    "news_sentiment": {"train": 450, "val": 100, "test": 120},
    "headlines": {"train": 150, "val": 40, "test": 50},
}
TRACES_MAX = 2000
# news_event_small: the same bundle with only this many train rows per label (v1 "little data"
# against v2 "all data"); test, config and job description are identical to news_event.
SMALL_PER_LABEL = 10
SEED = 13

CONFIGS = {
    "news_event": {"student": "Qwen3.5-0.8B", "teacher": "zai.glm-5.3-low-thinking", "train_target": 3000},
    "news_sentiment": {"student": "Qwen3.5-0.8B", "teacher": "zai.glm-5.3-low-thinking", "train_target": 2000},
    "headlines": {"student": "Qwen3.5-0.8B", "teacher": "zai.glm-5.3-low-thinking", "train_target": 3000},
    "cases": {"student": "Qwen3.5-2B", "teacher": "zai.glm-5.3-low-thinking", "train_target": 2000},
}


def balanced(rows: list[dict], cap: int, rng: random.Random) -> list[dict]:
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    picked = []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        picked += group[:cap]
    return sorted(picked, key=lambda r: r["id"])


def messages(row: dict) -> dict:
    return {"messages": [{"role": "user", "content": row["input"]}, {"role": "assistant", "content": row["label"]}]}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def config_yaml(name: str, task: tasks.Task) -> str:
    cfg = CONFIGS[name]
    noun = "pair outlier case" if name == "cases" else "headline"
    values = "\n".join(f'        - "a {noun} whose correct label is {lab}"' for lab in task.labels)
    return f"""# distil labs job config for the {name} task (docs: reference/config).
# A fixed label set is a question-answering task; the labels live in job_description.json.
base:
  task: question-answering
  student_model_name: {cfg["student"]}
  teacher_model_name: {cfg["teacher"]}
evaluation:
  llm_as_a_judge_model_name: zai.glm-5.3-low-thinking
synthgen:
  train_generation_target: {cfg["train_target"]}
  mutators:
    - name: label
      description: which label the example has
      method: adaptive
      values:
{values}
trace_processing:
  observation_format: openai_messages
"""


NEWS_SYNTH = (
    "Inputs are short English stock market news headlines or news tweets as posted by outlets such as Reuters, "
    "CNBC, Seeking Alpha, Bloomberg or company press releases, 5 to 40 words, often with $TICKER cashtags and "
    "hashtags, about US and international listed companies and markets. Vary companies, sectors, outlets and "
    "phrasing; include ambiguous and borderline headlines, not only clear-cut ones."
)

SYNTH_INSTRUCTIONS = {
    "news_event": NEWS_SYNTH,
    "news_sentiment": NEWS_SYNTH,
    "headlines": (
        "Inputs are single English financial news headlines as published by news wires and finance sites "
        "(Benzinga, Zacks, Reuters, Motley Fool style), 5 to 25 words, about large US companies. "
        "Company names are anonymized as Company A or Company B, years as [year] and dates as [date]; other "
        "companies and tickers may appear. Vary publishers, tone and phrasing; include lists, analyst notes, "
        "earnings previews and ambiguous headlines, not only clear-cut ones."
    ),
    "cases": (
        "Inputs follow exactly the line format of the train rows: spread statistics of an anonymized stock pair "
        "on the day its z-score crossed 2, volume spikes, SEC 8-K items and the latest headline. Keep numbers "
        "realistic (half-life 3 to 60 days, correlation 0.3 to 0.95, p-values 0 to 1)."
    ),
}


def job_description(task: tasks.Task) -> dict:
    return {
        "task_description": task.system_prompt,
        "synthetic_data_generation_instructions": SYNTH_INSTRUCTIONS[task.name],
        "llm_as_a_judge_instructions": (
            "The answer is correct only if it is exactly one of the listed labels, with no other text, "
            "and it is the label the definitions assign to the input. Any other answer is wrong."
        ),
    }


def export(name: str) -> dict[str, list[dict]]:
    task = tasks.get(name)
    rng = random.Random(SEED)
    rows = tasks.load_examples(name)
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    if name in CAPS:
        sets = {s: balanced(by_split[s], CAPS[name][s], rng) for s in by_split}
        traces = []
        if name == "headlines":
            used = {r["id"] for r in sets["train"]}
            rest = [r for r in by_split["train"] if r["id"] not in used]
            rng.shuffle(rest)
            traces = rest[:TRACES_MAX]
    else:
        sets, traces = by_split, []

    for split, rs in sets.items():
        write_jsonl(config.OUT / "sets" / f"{name}_{split}.jsonl", rs)
    if name == "headlines":
        # Headlines attached to test-period cases, for backtest variant c (never trained on).
        case_test = [r for r in by_split["test"] if r["meta"]["in_case"]]
        write_jsonl(config.OUT / "sets" / "headlines_case_test.jsonl", case_test)
    out = config.OUT / "distil" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(config_yaml(name, task))
    (out / "job_description.json").write_text(json.dumps(job_description(task), indent=2) + "\n")
    write_jsonl(out / "train.jsonl", [messages(r) for r in sets["train"]])
    write_jsonl(out / "test.jsonl", [messages(r) for r in sets["test"]])
    if traces:
        # No system turn: with variable_system_prompt off, distil labs uses task_description.
        write_jsonl(out / "traces.jsonl", [messages(r) for r in sorted(traces, key=lambda r: r["id"])])
    elif (out / "traces.jsonl").exists():
        (out / "traces.jsonl").unlink()
    if name == "news_event":
        small = out.parent / f"{name}_small"
        small.mkdir(parents=True, exist_ok=True)
        for f in ("config.yaml", "job_description.json", "test.jsonl"):
            (small / f).write_text((out / f).read_text())
        write_jsonl(small / "train.jsonl", [messages(r) for r in balanced(sets["train"], SMALL_PER_LABEL, random.Random(SEED))])
    return {**sets, "traces": traces}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=[*tasks.TASK_NAMES, "all"], default="all")
    args = parser.parse_args()
    names = [n for n in tasks.TASK_NAMES if n != "headlines" or (config.OUT / "headlines.jsonl").exists()]
    names = [n for n in names if n != "cases" or (config.OUT / "cases.jsonl").exists()]
    for name in names if args.task == "all" else [args.task]:
        sets = export(name)
        print(f"[{name}] -> {config.OUT / 'distil' / name}")
        for split in ("train", "val", "test"):
            counts = Counter(r["label"] for r in sets[split])
            print(f"  {split:5s} {len(sets[split]):5d}  {dict(sorted(counts.items()))}")
        if sets["traces"]:
            print(f"  traces {len(sets['traces'])}")


if __name__ == "__main__":
    main()
