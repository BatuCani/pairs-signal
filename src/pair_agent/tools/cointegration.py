import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import CollinearityWarning
from statsmodels.tsa.stattools import coint

from pair_agent.models.schemas import TestResult


def calculate_cointegration(prices: pd.DataFrame, threshold: float = 0.05) -> TestResult:
    try:
        if len(prices) < 30 or prices.shape[1] != 2 or prices.isna().any().any():
            raise ValueError("need at least 30 aligned complete observations")
        with warnings.catch_warnings():
            warnings.simplefilter("error", CollinearityWarning)
            stat, p_value, _ = coint(prices.iloc[:, 0], prices.iloc[:, 1], trend="c", autolag="aic")
        if not np.isfinite([stat, p_value]).all():
            raise ValueError("degenerate cointegration result")
        return TestResult(test_statistic=float(stat), p_value=float(p_value),
                          passes_threshold=bool(p_value < threshold))
    except (ValueError, np.linalg.LinAlgError, CollinearityWarning) as exc:
        return TestResult(error=str(exc))
