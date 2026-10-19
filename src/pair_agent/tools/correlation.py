import numpy as np
import pandas as pd

from pair_agent.data.market_data import DataError
from pair_agent.models.schemas import CorrelationResult


def calculate_return_correlation(prices: pd.DataFrame) -> CorrelationResult:
    returns = prices.pct_change(fill_method=None).dropna()
    if len(returns) < 3 or returns.shape[1] != 2:
        raise DataError("insufficient returns for correlation")
    correlation = returns.iloc[:, 0].corr(returns.iloc[:, 1])
    if not np.isfinite(correlation):
        raise DataError("undefined return correlation")
    return CorrelationResult(correlation=float(correlation), n_observations=len(returns))
