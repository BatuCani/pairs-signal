"""Upload the fixed train/val/test sets as Langfuse datasets pairs-<task>-<split>.

Item ids are deterministic (<task>-<example id>), so re-running updates items in place.
Needs LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST; --dry-run only counts.
"""

import argparse
import json

from pairs_signal import config, tasks
from pairs_signal.llm import langfuse_enabled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*tasks.TASK_NAMES, "all"], default="all")
    parser.add_argument("--splits", default="train,val,test,case_test,phrasebank_test")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    names = tasks.TASK_NAMES if args.task == "all" else [args.task]
    if not args.dry_run and not langfuse_enabled():
        raise SystemExit("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set (use --dry-run to only count)")
    lf = None
    if not args.dry_run:
        from langfuse import get_client

        lf = get_client()
    for name in names:
        task = tasks.get(name)
        for split in args.splits.split(","):
            path = config.OUT / "sets" / f"{name}_{split}.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(line) for line in path.open()]
            dataset = f"pairs-{name}-{split}"
            print(f"{dataset}: {len(rows)} items" + (" (dry run)" if args.dry_run else ""))
            if lf is None:
                continue
            lf.create_dataset(
                name=dataset,
                description=f"{name} {split} split; fixed test set never used for training" if split == "test" else f"{name} {split} split",
                metadata={"labels": list(task.labels), "system_prompt": task.system_prompt},
            )
            for r in rows:
                lf.create_dataset_item(
                    dataset_name=dataset,
                    id=f"{name}-{r['id']}",
                    input={"text": r["input"]},
                    expected_output={"label": r["label"]},
                    metadata={"id": r["id"], "split": split, **r["meta"]},
                )
    if lf is not None:
        lf.flush()


if __name__ == "__main__":
    main()
