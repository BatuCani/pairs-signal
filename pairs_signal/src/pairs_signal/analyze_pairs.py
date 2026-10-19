"""Engle-Granger cointegration, hedge ratio, spread and rolling z-score per pair.

Everything that feeds a trading decision on day t is estimated from data up to t only:
hedge ratio, intercept and the cointegration test are refit every REFIT_DAYS on the
trailing FORMATION_DAYS window and then frozen for the following block of days.

Outputs:
  data/interim/pairs_summary.csv   one row per pair (full-sample and rolling statistics),
                                   also copied to data/out/pairs_summary.csv
  data/interim/spreads.parquet     daily spread, z-score and fit parameters per pair
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from pairs_signal import config

REFIT_DAYS = 21


def load_log_prices() -> pd.DataFrame:
    prices = pd.read_csv(config.PRICES_FILE, index_col="date", parse_dates=True)
    return np.log(prices)


def hedge_fit(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS y = alpha + beta * x on log prices."""
    res = sm.OLS(y, sm.add_constant(x)).fit()
    return float(res.params[0]), float(res.params[1])


def half_life(spread: np.ndarray) -> float:
    """Half-life in days from an AR(1) fit of the spread changes; inf if not mean reverting."""
    lagged = spread[:-1]
    delta = np.diff(spread)
    res = sm.OLS(delta, sm.add_constant(lagged)).fit()
    b = float(res.params[1])
    return float(-np.log(2) / b) if b < 0 else float("inf")


def rolling_pair(log_a: pd.Series, log_b: pd.Series) -> pd.DataFrame:
    """Daily spread/z-score with parameters refit on trailing windows (no lookahead)."""
    n = len(log_a)
    ya, xb = log_a.to_numpy(), log_b.to_numpy()
    ret_a, ret_b = log_a.diff().to_numpy(), log_b.diff().to_numpy()
    rows = []
    coint_streak = 0
    for start in range(config.FORMATION_DAYS, n, REFIT_DAYS):
        lo = start - config.FORMATION_DAYS
        y, x = ya[lo:start], xb[lo:start]
        alpha, beta = hedge_fit(y, x)
        pvalue = float(coint(y, x)[1])
        coint_streak = coint_streak + REFIT_DAYS if pvalue < config.COINT_PVALUE else 0
        form_spread = y - alpha - beta * x
        hl = half_life(form_spread)
        corr = float(pd.Series(ret_a[lo + 1 : start]).corr(pd.Series(ret_b[lo + 1 : start])))
        end = min(start + REFIT_DAYS, n)
        # Spread under the frozen parameters, with ZSCORE_DAYS of history for the rolling stats.
        hist = max(0, start - config.ZSCORE_DAYS)
        spread = pd.Series(ya[hist:end] - alpha - beta * xb[hist:end])
        mu = spread.rolling(config.ZSCORE_DAYS).mean()
        sd = spread.rolling(config.ZSCORE_DAYS).std()
        for i in range(start, end):
            j = i - hist
            rows.append(
                {
                    "date": log_a.index[i],
                    "spread": spread.iloc[j],
                    "mu": mu.iloc[j],
                    "sd": sd.iloc[j],
                    "z": (spread.iloc[j] - mu.iloc[j]) / sd.iloc[j],
                    "alpha": alpha,
                    "beta": beta,
                    "coint_p": pvalue,
                    "coint_days": coint_streak,
                    "half_life": hl,
                    "corr": corr,
                    "ret_a": ret_a[i],
                    "ret_b": ret_b[i],
                    "refit_start": log_a.index[start],
                }
            )
    return pd.DataFrame(rows).set_index("date")


def analyze() -> tuple[pd.DataFrame, pd.DataFrame]:
    logp = load_log_prices()
    summaries, frames = [], []
    for a, b in config.PAIRS:
        pair = f"{a}/{b}"
        la, lb = logp[a], logp[b]
        full_p = float(coint(la, lb)[1])
        _, full_beta = hedge_fit(la.to_numpy(), lb.to_numpy())
        roll = rolling_pair(la, lb)
        windows = roll.drop_duplicates("refit_start")
        share = float((windows["coint_p"] < config.COINT_PVALUE).mean())
        summaries.append(
            {
                "pair": pair,
                "a": a,
                "b": b,
                "full_coint_p": round(full_p, 4),
                "full_hedge_ratio": round(full_beta, 3),
                "corr_returns": round(float(la.diff().corr(lb.diff())), 3),
                "rolling_coint_share": round(share, 3),
                "median_half_life": round(float(windows["half_life"].replace(np.inf, np.nan).median()), 1),
                "days_abs_z_ge_2": int((roll["z"].abs() >= config.ENTRY_Z).sum()),
                "status": "core" if full_p < config.PAIR_PVALUE else "illustrative" if full_p < config.ILLUSTRATIVE_PVALUE else "excluded",
            }
        )
        frames.append(roll.assign(pair=pair, a=a, b=b))
    summary = pd.DataFrame(summaries).sort_values("full_coint_p")
    spreads = pd.concat(frames).reset_index()
    return summary, spreads


def main() -> None:
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    summary, spreads = analyze()
    summary.to_csv(config.INTERIM / "pairs_summary.csv", index=False)
    config.OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.OUT / "pairs_summary.csv", index=False)
    spreads.to_parquet(config.INTERIM / "spreads.parquet", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False))
    counts = summary["status"].value_counts().to_dict()
    print(f"\n{counts} (core: p < {config.PAIR_PVALUE}; illustrative: not significant, p < {config.ILLUSTRATIVE_PVALUE})")


if __name__ == "__main__":
    main()
