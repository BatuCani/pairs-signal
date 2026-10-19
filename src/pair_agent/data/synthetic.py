"""Seeded artificial data for demonstrations, never presented as market evidence."""
import numpy as np
import pandas as pd

from pair_agent.data.universe import default_universe


def synthetic_prices(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", "2026-10-16")
    n = len(dates)
    common = 100 + np.cumsum(rng.normal(0.025, 0.65, n))
    residual = np.zeros(n)
    for i in range(1, n):
        residual[i] = 0.82 * residual[i - 1] + rng.normal(0, 0.65)
    prices = {}
    for stock in default_universe():
        prices[stock.ticker] = 100 * np.exp(np.cumsum(rng.normal(0.00015, 0.012, n)))
    prices["PEP"] = common
    prices["KO"] = 12 + 0.75 * common + residual
    # Correlated increments, independent stochastic trend: not cointegrated.
    prices["NVDA"] = 120 + np.cumsum(rng.normal(0.03, 0.7, n))
    prices["AMD"] = 25 + prices["NVDA"] + np.cumsum(rng.normal(0, 0.4, n))
    return pd.DataFrame(prices, index=dates)
