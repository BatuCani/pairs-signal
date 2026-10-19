# Pair Research Agent

An AI-assisted quantitative research MVP for finding potentially mean-reverting
equity pairs. **Agents decide what to investigate; deterministic Python tools
perform every calculation.** The complete baseline works offline without an LLM.

This is not financial advice, a live trading system, or a broker integration.
It cannot place orders. Backtests do not guarantee future performance.

## Quick start

Python 3.11+ is required. Open this directory as a project in PyCharm. On this
machine a Python 3.12 virtual environment is already installed in `.venv`.
Select `.venv/Scripts/python.exe` as the PyCharm project interpreter.

From the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pair_agent.cli research --ticker KO --synthetic
.\.venv\Scripts\python.exe -m pair_agent.cli research --ticker NVDA
.\.venv\Scripts\python.exe scripts/run_evaluation.py --synthetic
.\.venv\Scripts\python.exe -m pytest -q
```

On a fresh machine (macOS/Linux: use `python3` and `.venv/bin/python`):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Alternatively, `uv venv` followed by `uv pip install -e ".[dev]"` installs the
project. `requirements-lock.txt` records exact versions used during verification;
install it before `pip install --no-deps -e .` to reproduce those dependencies.

The artificial demo deliberately includes KO/PEP cointegration, NVDA/AMD correlated
but non-cointegrated prices, and unrelated series. Its prices and scores are
**invented fixtures, not real evidence about those companies**. Synthetic mode is
always explicitly requested and never silently replaces failed market downloads.

## Architecture

```text
ResearchRequest
       |
ResearchAgent (heuristic or LLM tool loop)
       |
Economic peer filter -> training return-correlation screen
       |
Deterministic Quant Tools -> PairAnalysis / validation backtest
       |
Accept or reject -> freeze ordered selection -> held-out diagnostic
       |
Local JSONL Trace -> deterministic Evaluator -> scored dataset export

Future (adapters, not installed integrations):
Local Trace -> Langfuse
Langfuse successes/failures -> Distil Labs -> Autoresearch
Trained model -> Runpod -> ModelClient -> Agent v2 -> Langfuse
```

* `data/`: provider interface, adjusted-price CSV cache, synthetic/local provider,
  replaceable universe of 46 curated US stocks across several sectors.
* `tools/`: correlation, Engle-Granger, OLS, residual ADF, AR(1) half-life and ledger.
* `research/`: chronological partitioning, economic peers, analyzer and quality score.
* `agent/`: common workflow, deterministic baseline and optional LLM investigation loop.
* `tracing/`: provider-independent `TraceSink` and `LocalJsonlTraceSink`.
* `evaluation/`: objective scores, failure categories, benchmark and dataset export.
* `integrations/`: a real OpenAI-compatible wire client and documented future boundaries.

Each request writes `data/traces/<trace_id>.jsonl`. CLI research also writes a typed
report to `data/processed/research-<trace_id>.json`. Traces include configuration,
data fingerprints for training/validation, period boundaries, tool inputs/outputs,
model messages, failures, model/version, latency and evaluation scores. API keys and
HTTP authorization headers are never trace fields. One sink instance is intended
for sequential use; create one per concurrent worker. Completed files are written
atomically; this MVP does not implement distributed tracing or process-crash recovery.

## Time-series and statistical discipline

The default date window starts 2022-01-01 and ends tomorrow (exclusive). A fixed
business-day calendar partitions it into 60% training, 20% validation and 20% test.
These boundaries are chosen **before reading any prices**. Missing future prices
cannot move the training boundary. This provides shorter datasets with chronological
fractional periods; minimum observation gates still apply. No random time split exists.

Use the supplied explicit calendar split when comparing repeated experiments:

```powershell
.\.venv\Scripts\python.exe -m pair_agent.cli research --ticker KO --config configs/calendar.example.json
```

That example trains through 2024, validates in 2025, and reserves 2026 through
2026-10-19 as held out. Edit dates intentionally when changing the experiment.

Selection tools download only training or training+validation windows. Training
return correlation ranks peers. OLS fits `A = intercept + beta * B + residual`;
beta, intercept, residual mean and standard deviation are fitted on training only.
Frozen parameters are applied unchanged to validation and then the final test.
Thresholds and transaction costs are configuration inputs, never optimized on test.

The reported **train backtest** is a causal diagnostic: fit on the first 60% of the
training block and replay its remaining 40%. It does not reuse full-training
parameters in an earlier historical replay. Its periods are saved separately.
This is a simpler MVP than rolling walk-forward estimation; train-to-validation
comparison uses different calibration lengths and is therefore only a diagnostic.

Acceptance and ranking use validation and training evidence only. The selected
ordered list is frozen first; only selected accepted pairs receive a held-out
backtest afterward. Test results never change acceptance, rank or the evaluation
optimization target. A poor test is still reported. Missing test observations
produce `test_error`, rather than changing the prior decision. Repeatedly examining
the same test while developing would contaminate it; reserve a fresh final holdout
for a real experiment.

Engle-Granger p-values supply the cointegration gate. Residual ADF is an additional
diagnostic: its ordinary ADF p-value is **not** a second independent calibrated
cointegration p-value, because OLS residuals were estimated. Engle-Granger assumes
I(1) inputs; this MVP does not separately verify integration order. Half-life uses
the exact discrete AR(1) formula `-log(2)/log(phi)` for `0 < phi < 1`; other roots
return null. Constant or near-collinear degenerate inputs are rejected.

## Backtest conventions

* A close-of-day z-score above +2 requests a short spread; below -2 a long spread.
* A signal observed at close `t` executes at close `t+1`, never earlier. The new
  position earns P&L only after that execution. No same-close perfect fills are assumed.
* Exit when `abs(z) < 0.5` or the spread crosses its mean; no instant flip on exit.
* At entry, share quantities are proportional to `[1, -beta]` and sized to 1x gross
  current equity. Shares remain fixed until exit; both legs' actual notionals are
  charged proportional costs. Default: 10 basis points per traded gross notional.
* Each period starts flat, ends with charged liquidation, and compounds from 1.0.
  A completed round trip (including final liquidation) counts as one trade; win
  rate includes entry, exit and accrued borrow costs. `total_cost` is in units of
  initial period equity, and turnover is summed traded notional / pre-bar equity.
* Optional annual short-borrow cost is configurable; default zero. Metrics assume
  252 observations/year, risk-free return zero and closing adjusted prices.
* OLS share hedging is **not guaranteed dollar or market-beta neutrality**. The
  ledger records net exposure and metrics report mean absolute net exposure.
  Equity exhaustion raises an explicit data/backtest error.

No order book, liquidity constraints, locate availability, dividends paid on shorts
separately from adjusted prices, margin mechanics, market impact, funding-rate model
or corporate-action point-in-time replay is implemented. An explicit borrow rate
and doubled trading-cost stress improve transparency without pretending to model
all execution frictions. Adjusted historical prices can be revised by Yahoo.

## Data and universe

Yahoo data uses explicit `auto_adjust=False` and requires `Adj Close`; missing
adjustments are never silently replaced by raw close. Date windows use exclusive
end dates. Per-symbol/window CSV snapshots avoid repeated downloads; `--refresh`
requests new data, including vendor revisions. Cache and yfinance metadata are
stored under the project's `data/raw`. Preserve raw snapshots for reproducibility.

Dates are normalized, sorted and deduplicated. Nonfinite/nonpositive prices are
invalid. A pair is aligned separately in each period without forward filling;
more than 5% missing aligned observations or a gap over 5 business days rejects
the period. Minimums: 126 training and 40 evaluation observations. Holidays and
short allowed gaps mean observations are an approximation to daily sessions.
Unavailable symbols are reported explicitly; failed downloads never fabricate data.

The curated universe is a current convenience sample, not a survivorship-free,
point-in-time security master. Metadata is intentionally coarse. Peers are from
the same sector, with same-industry preference before return-correlation screening;
correlation ranks candidates for bounded deeper investigation (12 by default).
No liquidity filter is implemented. To replace the universe, pass `--universe`
a JSON array of `{ "ticker": "KO", "sector": "Consumer Staples", "industry": "Beverages" }`.

Use your own adjusted-price CSV (first column dates, subsequent columns tickers):

```powershell
.\.venv\Scripts\python.exe -m pair_agent.cli research --tickers KO PEP KDP --top-k 3 --csv prices.csv
.\.venv\Scripts\python.exe -m pair_agent.cli research --request "Find three pairs among major US banks."
```

The heuristic text parser supports known uppercase ticker mentions, common sector
keywords and simple requested counts. Use explicit flags for arbitrary wording.

## Scoring, rejection and evaluation

Pair-quality weights: 25% statistical validity, 25% validation performance, 20%
train-to-validation stability, 15% drawdown, 10% doubled-cost robustness, 5% trade
activity. Components are separately exposed. These scores are transparent engineering
heuristics in [0,1], not probabilities or a calibrated forecast of profitability.

Hard acceptance gates require sufficient data, positive beta, minimum return
correlation, cointegration and residual-stationarity passes, sensible half-life,
positive validation returns and Sharpe, sufficient completed trades, acceptable
drawdown, positive doubled-cost returns and minimum total score. Large degradation
also rejects. Configurations are validated; see `ResearchConfig` for all fields.

Failure categories distinguish correlation without cointegration, nonstationarity,
bad half-life, overfitting, costs, drawdown, missing data, excessive tools, model
errors and forced selections. A model cannot override deterministic acceptance.

```powershell
.\.venv\Scripts\python.exe scripts/run_evaluation.py --synthetic
.\.venv\Scripts\python.exe scripts/run_evaluation.py --tasks data/tasks.jsonl --limit 3
```

Twenty starter tasks cover single targets, related groups, unrelated pairs,
missing symbols and no-peer requests. Evaluation writes per-task results, traces,
the full configuration and aggregate metrics under `data/evaluations/`. The console
prints an agent summary for comparisons. `--agent llm` evaluates a configured LLM
instead. The deterministic evaluator includes pair quality, statistical validity,
cointegration, stationarity, out-of-sample quality, robustness and tool efficiency.
A justified abstention receives at most 0.5; lack of evidence receives zero.
There are no ground-truth profitability labels or claims of benchmark accuracy.

## Optional LLM / future Runpod deployment

Heuristic mode is the default and requires no credentials. `--agent auto` falls
back to it if no `LLM_API_KEY` exists; `--agent llm` requires both key and model.
`.env.example` documents variables but is **not automatically loaded**:

```powershell
$env:LLM_BASE_URL = "https://your-compatible-endpoint/v1"
$env:LLM_API_KEY = "your-key"
$env:LLM_MODEL = "your-served-model"
.\.venv\Scripts\python.exe -m pair_agent.cli research --ticker KO --agent llm
```

The real bounded tool loop lets the model pick individual statistical tools, order
investigations and finish with an empty or ordered selection. Pair scope, argument
schemas, deep-analysis budgets and deterministic acceptance are enforced in Python.
The final report always uses tool result objects, not a model-authored numeric
narrative. Model interactions are traced without hidden chain-of-thought requests.

The client uses the Chat Completions function-calling protocol, with no provider
SDK dependency. To use Runpod later, swap the base URL, model and key for a compatible
deployment. Compatibility depends on the server and model supporting tool calls;
models requiring Responses API need a separate `ModelClient` adapter. Live LLM
connectivity is not part of offline verification. Never put credentials in source.

## Dataset preparation and integration TODOs

```powershell
.\.venv\Scripts\python.exe scripts/generate_dataset.py --min-score 0.8
.\.venv\Scripts\python.exe scripts/generate_dataset.py --min-score 0 --max-score 0.5 --failures-only --output data/processed/failures.jsonl
```

Exports contain user/assistant messages and trace IDs, source, score, configuration,
failure modes and actual tool/model steps. Python APIs additionally expose
`select_traces`, `export_training_dataset` and `export_failure_dataset`. Filter score
ranges or failed examples deliberately. Failed examples must not be treated as
positive training targets. Synthetic source identity stays in metadata.

No model is trained and no Langfuse/Distil/Autoresearch APIs are invented.
See `src/pair_agent/integrations/README.md` for `LangfuseTraceSink`,
`DistilDatasetExporter`, Autoresearch and Runpod adapter boundaries. Future work:
point-in-time data and I(1) checks, multiple-testing correction, rolling walk-forward
evaluation, uncertainty estimates and a more realistic execution/financing model.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
```

Tests use seeded fixtures and mocked provider/model responses, not live APIs. They
cover return correlation, correlated/non-cointegrated versus cointegrated examples,
OLS and spread, ADF, half-life, date splitting, leakage, exact share P&L and costs,
next-close execution, initial-capital drawdown, ranking/rejection, held-out isolation,
typed trace round trips, evaluation, exports, cache and CLI. LLM tests run actual
deterministic tools through a scripted model boundary without API calls.

Reference implementations follow the [statsmodels Engle-Granger documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html),
[ADF documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html),
[yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html),
and [OpenAI function-calling protocol](https://developers.openai.com/api/docs/guides/function-calling).

## Optional chart

The optional plotting extra is installed in this machine's environment. On a
fresh environment, install `pip install -e ".[plots]"`. Then run:

```powershell
.\.venv\Scripts\python.exe scripts/plot_pair.py KO PEP --synthetic
```

The output is `data/processed/equity.png`, clearly labeled as synthetic.
See `VERIFICATION.md` for the completed local checks and remaining integrations.
