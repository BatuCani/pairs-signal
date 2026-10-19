"""The model tasks: their prompts, label sets and answer parsing.

news_event (product): stock news headline -> one of 10 event types (public labeled data).
news_sentiment (product, optional): stock news headline -> positive / negative / neutral.
headlines (pairs extra): FNSPID pair-firm headline -> one of 8 event labels.
cases (pairs extra): pair outlier with metrics and headline -> reverts / no_revert.
"""

import json
import re
from dataclasses import dataclass
from functools import cache

from pairs_signal import config

CASE_LABELS = {
    "reverts": f"The spread z-score returns inside |z| < {config.EXIT_Z} within {config.HORIZON_DAYS} trading days after the outlier day (hedge ratio, spread mean and spread std frozen on the outlier day).",
    "no_revert": f"The spread z-score stays at |z| >= {config.EXIT_Z} for all of the next {config.HORIZON_DAYS} trading days.",
}
CASE_DESCRIPTION = (
    "You judge pairs-trading outliers. Two historically related stocks (anonymized as Company A and "
    f"Company B) have a spread whose z-score just crossed |z| >= {config.ENTRY_Z}. You get spread statistics "
    "and the latest headline about either company. Decide whether the spread will revert. "
    "Headlines about fundamental events (earnings shocks, M&A, legal or regulatory action, management changes) "
    "make reversion less likely; noise and market-wide news make it more likely."
)


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    labels: dict[str, str]

    @property
    def system_prompt(self) -> str:
        """Task description, label definitions and the answer rule, as distil labs expects."""
        defs = "\n".join(f"- {k}: {v}" for k, v in self.labels.items())
        return f"{self.description}\n\nLabels:\n{defs}\n\nAnswer with the label only."

    def parse(self, text: str) -> str | None:
        """First label mentioned in a model answer (tolerates JSON, quotes, reasoning tags)."""
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                text = str(obj.get("label") or obj.get("decision") or text)
        except ValueError:
            pass
        hits = [(m.start(), lab) for lab in self.labels for m in re.finditer(rf"\b{re.escape(lab)}\b", text)]
        return min(hits)[1] if hits else None


@cache
def headline_task() -> Task:
    spec = json.loads((config.TASKS / "headline_labels.json").read_text())
    return Task("headlines", spec["task_description"].replace(" Answer with the label only.", ""), spec["labels"])


@cache
def case_task() -> Task:
    return Task("cases", CASE_DESCRIPTION, CASE_LABELS)


@cache
def news_task(name: str) -> Task:
    spec = json.loads((config.TASKS / f"{name}_labels.json").read_text())
    return Task(name, spec["task_description"], spec["labels"])


TASK_NAMES = ["news_event", "news_sentiment", "headlines", "cases"]


def get(name: str) -> Task:
    if name.startswith("news_"):
        return news_task(name)
    return {"headlines": headline_task, "cases": case_task}[name]()


@cache
def shock_labels() -> frozenset[str]:
    spec = json.loads((config.TASKS / "headline_labels.json").read_text())
    return frozenset(spec["fundamental_shock_labels"])


def load_examples(task: str) -> list[dict]:
    """All labeled examples of a task as {id, split, input, label, meta} dicts."""
    if task.startswith("news_"):
        rows = [json.loads(line) for line in (config.OUT / "news" / f"{task}.jsonl").open()]
        return [
            {"id": r["id"], "split": r["split"], "input": r["text"], "label": r["label"], "meta": {"source": r["source"], "source_label": r["source_label"]}}
            for r in rows
        ]
    if task == "headlines":
        rows = [json.loads(line) for line in (config.OUT / "headlines.jsonl").open()]
        return [
            {"id": r["headline_id"], "split": r["split"], "input": r["text"], "label": r["label"], "meta": {"label_source": r["label_source"], "in_case": r["in_case"]}}
            for r in rows
        ]
    rows = [json.loads(line) for line in (config.OUT / "cases.jsonl").open()]
    return [
        {"id": r["case_id"], "split": r["split"], "input": r["input"], "label": r["label"], "meta": {"pair_id": r["pair_id"], **r["outcome"]}}
        for r in rows
    ]
