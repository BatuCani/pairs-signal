import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from pair_agent.models.schemas import StationarityResult


def calculate_stationarity(spread: pd.Series, threshold: float = 0.05) -> StationarityResult:
    try:
        if len(spread) < 30 or not np.isfinite(spread).all() or spread.std() < 1e-10:
            raise ValueError("spread must be finite, nonconstant, and at least 30 observations")
        stat, p_value, *_ = adfuller(spread, regression="c", autolag="AIC")
        if not np.isfinite([stat, p_value]).all():
            raise ValueError("invalid ADF output")
        return StationarityResult(adf_statistic=float(stat), p_value=float(p_value),
                                  stationary=bool(p_value < threshold))
    except (ValueError, np.linalg.LinAlgError) as exc:
        return StationarityResult(error=str(exc))
