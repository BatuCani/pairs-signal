import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pandas as pd

from pair_agent.agent.heuristic_agent import HeuristicResearchAgent
from pair_agent.agent.llm_agent import LLMResearchAgent
from pair_agent.config import ResearchConfig
from pair_agent.data.market_data import FrameMarketData, YFinanceMarketData
from pair_agent.data.synthetic import synthetic_prices
from pair_agent.data.universe import load_universe
from pair_agent.evaluation.dataset import load_tasks
from pair_agent.evaluation.metrics import aggregate
from pair_agent.integrations.model_client import OpenAICompatibleModelClient
from pair_agent.models.schemas import ResearchRequest


def add_common_options(parser):
    parser.add_argument("--config", type=Path, help="JSON ResearchConfig overrides")
    parser.add_argument("--universe", type=Path, help="JSON stock metadata list")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--start")
    parser.add_argument("--end", help="exclusive end date")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true", help="explicit artificial offline demo")
    source.add_argument("--csv", type=Path, help="local wide adjusted-close CSV with date index")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--agent", choices=["heuristic", "llm", "auto"], default="heuristic")


def make_agent(args):
    overrides = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    for key in ["start", "end", "data_dir"]:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    overrides.setdefault("data_dir", os.getenv("PAIR_DATA_DIR", "data"))
    config = ResearchConfig.model_validate(overrides)
    if args.synthetic:
        provider = FrameMarketData(synthetic_prices(), source="synthetic-v1:seed42")
    elif args.csv:
        provider = FrameMarketData(pd.read_csv(args.csv, index_col=0, parse_dates=True), source="local-csv")
    else:
        provider = YFinanceMarketData(config.data_dir / "raw", refresh=args.refresh)
    universe = load_universe(args.universe)
    llm = args.agent == "llm" or (args.agent == "auto" and bool(os.getenv("LLM_API_KEY")))
    if llm:
        return LLMResearchAgent(config, provider, OpenAICompatibleModelClient.from_env(), universe)
    return HeuristicResearchAgent(config, provider, universe)


def render_result(result):
    lines = ["PAIR RESEARCH RESULT", f"Request: {result.request.request}",
             f"Agent: {result.agent_version}", f"Candidates screened/analyzed: {len(result.candidates)}",
             f"Decision: {result.decision}"]
    for pair in result.selected_pairs:
        lines += ["", f"Pair: {pair.key}", f"Return correlation: {pair.correlation:.4f}",
                  f"Cointegration p-value: {pair.cointegration_p_value:.6f}",
                  f"ADF p-value (residual diagnostic): {pair.adf_p_value:.6f}",
                  f"Hedge ratio: {pair.hedge_ratio:.4f}", f"Half-life: {pair.half_life_days:.2f} days",
                  f"Pair score: {pair.score * 100:.1f}/100"]
        for label in ["train", "validation", "validation_stressed", "test"]:
            metrics = getattr(pair, label)
            if metrics:
                lines.append(f"{label}: return={metrics.total_return:.2%}, Sharpe={metrics.sharpe_ratio:.2f}, "
                             f"drawdown={metrics.max_drawdown:.2%}, completed trades={metrics.number_of_trades}")
        if pair.test_error:
            lines.append("Held-out evaluation unavailable: " + pair.test_error)
    if not result.selected_pairs:
        counts = Counter(str(f) for p in result.candidates for f in p.failure_modes)
        lines += [f"  {reason}: {count}" for reason, count in counts.most_common()]
        if not counts:
            lines.append("No eligible peers in the requested universe, or no completed analysis.")
    if result.evaluation:
        lines.append(f"Evaluation overall: {result.evaluation.overall_score:.3f}")
    lines += [f"Trace ID: {result.trace_id}", f"Tool calls: {result.tool_call_count}"]
    return "\n".join(lines)


def run_evaluation(args):
    agent = make_agent(args)
    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    results, task_ids = [], []
    for i, task in enumerate(tasks, 1):
        request = ResearchRequest.model_validate({k: v for k, v in task.items() if k != "id"})
        result = agent.research(request)
        results.append(result)
        task_ids.append(task["id"])
        print(f"[{i:02}/{len(tasks):02}] {task['id']}: {result.decision} ({result.evaluation.overall_score:.3f})", flush=True)
    summary = aggregate(results, agent.config.p_threshold)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = agent.config.data_dir / "evaluations" / f"{agent.version}-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(agent=agent.version, source=getattr(agent.provider, "source", type(agent.provider).__name__),
                   config=agent.config.model_dump(mode="json"), summary=summary,
                   tasks=[dict(id=tid, **r.model_dump(mode="json")) for tid, r in zip(task_ids, results)])
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nAgent: {agent.version}")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pair research only; no real trade execution")
    sub = parser.add_subparsers(dest="command", required=True)
    research = sub.add_parser("research")
    add_common_options(research)
    research.add_argument("--ticker")
    research.add_argument("--tickers", nargs="+", default=[])
    research.add_argument("--request")
    research.add_argument("--top-k", type=int, default=1)
    evaluation = sub.add_parser("evaluate")
    add_common_options(evaluation)
    evaluation.add_argument("--tasks", type=Path, default=Path("data/tasks.jsonl"))
    evaluation.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            return run_evaluation(args)
        agent = make_agent(args)
        if args.synthetic:
            print("ARTIFICIAL SYNTHETIC DATA - results are not market evidence.\n")
        text = args.request or (f"Find {args.top_k} pair(s) for {args.ticker}." if args.ticker else
                                f"Find {args.top_k} pair(s) among {' '.join(args.tickers) or 'the stock universe'}.")
        request = ResearchRequest(request=text, ticker=args.ticker, tickers=args.tickers, top_k=args.top_k)
        result = agent.research(request)
        print(render_result(result))
        out = agent.config.data_dir / "processed" / f"research-{result.trace_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"Report: {out}")
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
