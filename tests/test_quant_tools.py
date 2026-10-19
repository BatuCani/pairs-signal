import numpy as np
import pandas as pd
import pytest

from pair_agent.data.market_data import DataError
from pair_agent.tools.cointegration import calculate_cointegration
from pair_agent.tools.correlation import calculate_return_correlation
from pair_agent.tools.half_life import estimate_half_life
from pair_agent.tools.spread import construct_spread, estimate_spread_parameters
from pair_agent.tools.stationarity import calculate_stationarity


def test_return_correlation_uses_returns():
    frame = pd.DataFrame({"a": [100, 110, 99, 108.9], "b": [50, 55, 49.5, 54.45]})
    result = calculate_return_correlation(frame)
    assert result.correlation == pytest.approx(1)
    assert result.n_observations == 3


def test_no_implicit_fill_in_returns():
    frame = pd.DataFrame({"a": [100, 110, np.nan, 120], "b": [50, 55, 60, 65]})
    with pytest.raises(DataError):
        calculate_return_correlation(frame)


def test_cointegrated_series_pass(prices):
    result = calculate_cointegration(prices[["KO", "PEP"]].iloc[:750])
    assert result.passes_threshold
    assert result.p_value < .05


def test_correlated_but_not_cointegrated(prices):
    frame = prices[["NVDA", "AMD"]].iloc[:750]
    assert calculate_return_correlation(frame).correlation > .7
    assert not calculate_cointegration(frame).passes_threshold


def test_unrelated_series(prices):
    assert abs(calculate_return_correlation(prices[["PFE", "MSFT"]]).correlation) < .15
    assert not calculate_cointegration(prices[["PFE", "MSFT"]]).passes_threshold


def test_degenerate_cointegration_fails():
    frame = pd.DataFrame({"a": np.arange(100), "b": np.arange(100) * 2})
    assert not calculate_cointegration(frame).passes_threshold


def test_hedge_and_spread(prices):
    frame = prices[["KO", "PEP"]].iloc[:750]
    params = estimate_spread_parameters(frame)
    assert params.hedge_ratio == pytest.approx(.75, abs=.05)
    spread = construct_spread(frame, params.hedge_ratio, params.intercept)
    assert abs(spread.mean()) < 1e-8
    assert calculate_stationarity(spread).stationary
    assert 1 < estimate_half_life(spread) < 10


def test_adf_rejects_constant_and_short_input():
    assert not calculate_stationarity(pd.Series(np.ones(100))).stationary
    assert calculate_stationarity(pd.Series([1, 2])).error


def test_half_life_invalid_cases():
    assert estimate_half_life(pd.Series(np.ones(100))) is None
    assert estimate_half_life(pd.Series(np.exp(np.arange(100) * .01))) is None


def test_discrete_half_life():
    rng = np.random.default_rng(7)
    values = [0.0]
    for _ in range(8000):
        values.append(.8 * values[-1] + rng.normal())
    assert estimate_half_life(pd.Series(values)) == pytest.approx(-np.log(2) / np.log(.8), rel=.12)
