"""Optional static chart from local research settings; no live-trading functionality."""
import argparse
from pathlib import Path

from pair_agent.cli import add_common_options, make_agent
from pair_agent.research.pair_analyzer import PairAnalyzer
from pair_agent.research.splitting import chronological_split
from pair_agent.tools.backtest import backtest_pair


def main():
    parser = argparse.ArgumentParser(description="Plot a validation ledger using training-only parameters")
    add_common_options(parser)
    parser.add_argument("ticker_a")
    parser.add_argument("ticker_b")
    parser.add_argument("--output", type=Path, default=Path("data/processed/equity.png"))
    args = parser.parse_args()
    agent = make_agent(args)
    analyzer = PairAnalyzer(agent.config, agent.provider)
    pair = analyzer.analyze_pair(args.ticker_a, args.ticker_b)
    if pair.parameters is None or pair.parameters.hedge_ratio <= 0:
        parser.exit(2, "No valid positive-hedge analysis to plot.\n")
    validation = analyzer.clean(chronological_split(analyzer.load(args.ticker_a, args.ticker_b), agent.config)["validation"],
                                agent.config.min_eval)
    ledger = backtest_pair(validation, pair.parameters, agent.config).ledger
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        parser.exit(2, 'Install the optional plots extra: pip install -e ".[plots]"\n')
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(ledger.index, ledger.equity)
    axes[0].set_ylabel("Equity (initial = 1)")
    axes[0].set_title(f"{pair.key} | validation | {'SYNTHETIC' if args.synthetic else 'adjusted prices'}")
    axes[1].plot(ledger.index, ledger.z_score)
    for level in [-agent.config.entry_z, agent.config.entry_z]:
        axes[1].axhline(level, color="gray", linestyle="--", linewidth=.8)
    axes[1].set_ylabel("Frozen-parameter z-score")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(args.output)


if __name__ == "__main__":
    main()
