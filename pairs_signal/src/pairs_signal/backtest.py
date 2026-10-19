"""Backtest the outlier cases with and without a headline filter.

Variants (each trade fades the spread at the event close, exits on |z| < EXIT_Z or after
HORIZON_DAYS, pays 2 x COST_BPS on gross notional):
  a_naive     trade every |z| >= ENTRY_Z event
  b_teacher   skip when the latest headline's teacher label is a fundamental_shock
  b_sec       skip when either firm filed an 8-K with a fundamental item before the close
              (real SEC data, not a model)
  c_student   skip when the small model labels the latest headline a fundamental_shock;
              needs --student-run (an experiment run over the case headlines), else n/a

Reports trades, hit rate (share of trades whose spread reverted), mean and summed net
return, and the base rate "always reverts" (= hit rate of a_naive).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from pairs_signal import config, tasks


def stats(df: pd.DataFrame, take: pd.Series) -> dict:
    trades = df[take]
    n = len(trades)
    return {
        "trades": n,
        "hit_rate": round(float((trades["label"] == "reverts").mean()), 4) if n else None,
        "mean_net_return": round(float(trades["net_return"].mean()), 6) if n else None,
        "sum_net_return": round(float(trades["net_return"].sum()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--student-run", default=None, help="data/out/runs/<run>.jsonl from experiment --model student --split case_test")
    args = parser.parse_args()

    cases = pd.read_csv(config.INTERIM / "cases_private.csv")
    if args.split != "all":
        cases = cases[cases["split"] == args.split]
    headlines = pd.DataFrame([json.loads(line) for line in (config.OUT / "headlines.jsonl").open()])
    shock = tasks.shock_labels()
    teacher = dict(zip(headlines["text"], headlines["label"]))
    source = headlines["label_source"].iloc[0] if len(headlines) else "none"

    teacher_shock = cases["headline"].map(lambda h: teacher.get(h) in shock if isinstance(h, str) else False)
    result = {
        "split": args.split,
        "cases": len(cases),
        "base_rate_reverts": round(float((cases["label"] == "reverts").mean()), 4) if len(cases) else None,
        "headline_labels": source,
        "variants": {
            "a_naive": stats(cases, pd.Series(True, index=cases.index)),
            "b_teacher": stats(cases, ~teacher_shock),
            "b_sec": stats(cases, ~cases["sec_fundamental"].astype(bool)),
        },
    }
    if args.student_run:
        rows = [json.loads(line) for line in Path(args.student_run).read_text().splitlines()]
        student = {r["input"]: r["predicted"] for r in rows}
        missing = int(cases["headline"].map(lambda h: isinstance(h, str) and h not in student).sum())
        student_shock = cases["headline"].map(lambda h: student.get(h) in shock if isinstance(h, str) else False)
        result["variants"]["c_student"] = {**stats(cases, ~student_shock), "headlines_without_prediction": missing}
    else:
        result["variants"]["c_student"] = "n/a: run experiment --model student --split case_test, then pass --student-run"

    config.OUT.mkdir(parents=True, exist_ok=True)
    (config.OUT / f"backtest_{args.split}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"split={args.split} cases={result['cases']} base rate reverts={result['base_rate_reverts']} headline labels={source}")
    print(f"{'variant':10s} {'trades':>6s} {'hit':>7s} {'mean ret':>9s} {'sum ret':>8s}")
    for name, v in result["variants"].items():
        if isinstance(v, str):
            print(f"{name:10s} {v}")
        else:
            hit = f"{v['hit_rate']:.1%}" if v["hit_rate"] is not None else "-"
            mean = f"{v['mean_net_return']:+.3%}" if v["mean_net_return"] is not None else "-"
            print(f"{name:10s} {v['trades']:6d} {hit:>7s} {mean:>9s} {v['sum_net_return']:+8.2%}")


if __name__ == "__main__":
    main()
