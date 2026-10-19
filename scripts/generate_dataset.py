import argparse
from pathlib import Path

from pair_agent.evaluation.dataset import export_training_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export scored completed traces, without training a model")
    parser.add_argument("--traces", type=Path, default=Path("data/traces"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/training.jsonl"))
    parser.add_argument("--min-score", type=float, default=.8)
    parser.add_argument("--max-score", type=float, default=1)
    parser.add_argument("--failures-only", action="store_true")
    args = parser.parse_args()
    count = export_training_dataset(args.traces, args.output, args.min_score, args.max_score, args.failures_only)
    print(f"Exported {count} traces to {args.output}")
