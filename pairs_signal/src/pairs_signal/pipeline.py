"""Run all offline data steps in order: download, pair analysis, cases, headline labels,
fixed sets / distil labs bundles, backtest.

  prepare            teacher labels via ANTHROPIC_API_KEY
  prepare --dry-run  keyword-rule labels, no API calls
"""

import argparse
import sys

from pairs_signal import (
    analyze_pairs,
    backtest,
    build_cases,
    export_distil,
    fetch_news,
    fetch_prices,
    label_headlines,
    sec_events,
)


def run(module, *argv: str) -> None:
    print(f"\n=== {module.__name__.split('.')[-1]} {' '.join(argv)}", flush=True)
    sys.argv = [module.__name__, *argv]
    module.main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="label headlines with keyword rules instead of the teacher")
    parser.add_argument("--pairs", choices=["core", "illustrative", "all"], default="core")
    args = parser.parse_args()
    run(fetch_prices)
    run(fetch_news)
    run(sec_events)
    run(analyze_pairs)
    run(build_cases, "--pairs", args.pairs)
    run(label_headlines, *(["--dry-run"] if args.dry_run else []))
    run(export_distil)
    run(backtest)


if __name__ == "__main__":
    main()
