# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.40", "langfuse>=3,<4", "rich>=13"]
# ///
"""News Gate terminal demo: classify test headlines live, then analyse the errors.

Uses the same model config, system prompt and call settings as ui/server.py.
Run: uv run ui/terminal_demo.py [--n 25]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server  # noqa: E402  (shares env, prompt, clients and mock rules)

console = Console(highlight=False)
OUT_PATH = server.UI_DIR / "data/next_train_errors.jsonl"


def load_test(n: int, seed: int) -> tuple[list[dict], int]:
    rows = []
    for line in (server.DATA_DIR / "test.jsonl").read_text().splitlines():
        msgs = json.loads(line)["messages"]
        rows.append({"headline": msgs[0]["content"], "expected": msgs[-1]["content"].strip()})
    total = len(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n], total


def header(total: int, n: int) -> None:
    if server.MOCK:
        model = "[bold yellow]MOCK[/] (keyword rules, no endpoint configured)"
    else:
        model = "\n".join(f"[bold]{k}[/]: {v['model']}  [dim]{v['base_url']}[/]" for k, v in server.VARIANTS.items())
    langfuse = "[green]on[/]" if server.LANGFUSE_ON and not server.MOCK else "[dim]off[/]"
    console.print(Panel(
        f"Model     {model}\nLangfuse  {langfuse}\nTest set  {total} headlines from {server.DATA_DIR.relative_to(server.REPO_DIR)}/test.jsonl, running {n}",
        title="[bold]News Gate[/] - real shock or just noise?", border_style="cyan",
    ))


def run(rows: list[dict]) -> list[dict]:
    results = []
    for i, row in enumerate(rows, 1):
        r = server.classify(row["headline"])[0]
        ok = r["label"] == row["expected"]
        mark = "[bold green]✓[/]" if ok else "[bold red]✗[/]"
        shock = "[red]SHOCK[/]" if r["shock"] else "[green]noise[/]"
        console.print(f"[dim]{i:>3}[/] {mark} [bold]{row['headline'][:90]}[/]")
        pred = r["label"] or f"[red]error: {r['error']}[/]"
        console.print(
            f"      predicted [bold {'green' if ok else 'red'}]{pred}[/]  dataset [cyan]{row['expected']}[/]  {shock}  [dim]{r['latency_ms']} ms[/]"
        )
        if r["trace_url"]:
            console.print(f"      [dim]-> Langfuse trace[/] [link={r['trace_url']}]{r['trace_url']}[/link]")
        results.append({**row, **r, "ok": ok})
    return results


def analyse(results: list[dict]) -> None:
    n = len(results)
    hits = sum(r["ok"] for r in results)
    lat = sorted(r["latency_ms"] for r in results)
    p95 = lat[min(n - 1, int(round(0.95 * (n - 1))))]
    console.rule("[bold]Error analysis")
    console.print(
        f"Accuracy [bold]{hits}/{n} = {100 * hits / n:.0f}%[/]   "
        f"latency mean [bold]{statistics.mean(lat):.0f} ms[/]  p95 [bold]{p95} ms[/]"
    )

    errors = [r for r in results if not r["ok"]]
    if not errors:
        console.print("[green]No errors.[/]")
        return
    pairs = Counter((r["expected"], r["label"] or "error") for r in errors)
    t = Table(title="Confusions (dataset -> predicted)", title_justify="left")
    t.add_column("dataset"); t.add_column("predicted"); t.add_column("count", justify="right")
    for (exp, pred), c in pairs.most_common():
        t.add_row(exp, pred, f"x{c}")
    console.print(t)

    per = {}
    for r in results:
        s = per.setdefault(r["expected"], [0, 0])
        s[0] += r["ok"]; s[1] += 1
    weak = sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][1])[:3]
    console.print("Weakest categories: " + ", ".join(f"[red]{k}[/] {v[0]}/{v[1]}" for k, v in weak))

    console.print("\n[bold]Example errors[/]")
    for r in errors[:3]:
        console.print(f"  [red]✗[/] {r['headline'][:110]}\n    dataset [cyan]{r['expected']}[/], predicted [red]{r['label']}[/]")


def next_loop(results: list[dict]) -> None:
    errors = [r for r in results if not r["ok"]]
    console.rule("[bold]Next loop step")
    with OUT_PATH.open("w") as f:
        for r in errors:
            f.write(json.dumps({"messages": [
                {"role": "user", "content": r["headline"]},
                {"role": "assistant", "content": r["expected"]},
            ]}) + "\n")
    rel = OUT_PATH.relative_to(server.REPO_DIR)
    console.print(f"Wrote [bold]{len(errors)}[/] misclassified headlines as training rows to [bold]{rel}[/] (distil format, dataset label as answer).")
    console.print("Feed them into the next training run (distil labs CLI; adjust names to your job):")
    console.print(f"  [bold cyan]distil dataset create news-event-v3 --train {rel} --job-description {server.DATA_DIR.relative_to(server.REPO_DIR)}/job_description.json[/]")
    console.print("  [bold cyan]distil generate-synthetic-data-train news-event-v3[/]  [dim]# synthetic variants around the error cases[/]")
    console.print("[dim]Langfuse -> distil labs -> Runpod -> Langfuse[/]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=25, help="number of test headlines (default 25)")
    ap.add_argument("--seed", type=int, default=7, help="shuffle seed for picking headlines")
    args = ap.parse_args()
    rows, total = load_test(args.n, args.seed)
    header(total, len(rows))
    results = run(rows)
    analyse(results)
    next_loop(results)


if __name__ == "__main__":
    main()
