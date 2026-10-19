import numpy as np
import pandas as pd

from pair_agent.data.market_data import DataError
from pair_agent.models.schemas import SpreadParameters


def construct_spread(prices: pd.DataFrame, hedge_ratio: float, intercept: float = 0) -> pd.Series:
    return (prices.iloc[:, 0] - hedge_ratio * prices.iloc[:, 1] - intercept).rename("spread")


def estimate_spread_parameters(train: pd.DataFrame) -> SpreadParameters:
    if len(train) < 30 or train.shape[1] != 2 or not np.isfinite(train).all().all():
        raise DataError("OLS requires complete training prices")
    x = train.iloc[:, 1].to_numpy()
    if x.std() < 1e-10:
        raise DataError("constant regressor")
    intercept, beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]),
                                     train.iloc[:, 0].to_numpy(), rcond=None)[0]
    spread = construct_spread(train, beta, intercept)
    std = float(spread.std(ddof=1))
    if not np.isfinite(std) or std < 1e-10:
        raise DataError("degenerate spread variance")
    return SpreadParameters(hedge_ratio=float(beta), intercept=float(intercept),
                            mean=float(spread.mean()), std=std,
                            fitted_until=train.index[-1].isoformat())
