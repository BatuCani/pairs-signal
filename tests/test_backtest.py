import pandas as pd
import pytest

from pair_agent.data.market_data import DataError
from pair_agent.models.schemas import SpreadParameters
from pair_agent.tools.backtest import backtest_pair


def parameters():
    return SpreadParameters(hedge_ratio=1, intercept=0, mean=0, std=1, fitted_until="2021-12-31")


def sample():
    return pd.DataFrame({"a": [100, 104, 106, 105, 102, 100], "b": [100] * 6},
                        index=pd.bdate_range("2022-01-03", periods=6))


def test_next_close_fills_and_exact_share_pnl(config):
    result = backtest_pair(sample(), parameters(), config.model_copy(update={"transaction_cost_bps": 0}))
    assert list(result.ledger.position) == [0, 0, -1, -1, -1, 0]
    assert result.ledger.pnl.iloc[2] == 0  # entry day doesn't earn the pre-entry move
    assert result.metrics.total_return == pytest.approx(6 / 206)
    assert result.metrics.number_of_trades == 1
    assert result.metrics.win_rate == 1


def test_costs_charge_entry_and_forced_liquidation(config):
    result = backtest_pair(sample(), parameters(), config)
    expected_cost = .001 + 200 / 206 * .001
    assert result.metrics.total_cost == pytest.approx(expected_cost)
    assert result.metrics.total_return == pytest.approx(6 / 206 - expected_cost)
    assert result.metrics.max_drawdown >= .001 - 1e-12
    stress = backtest_pair(sample(), parameters(), config, cost_multiplier=2)
    assert stress.metrics.total_return < result.metrics.total_return


def test_zero_positions_has_zero_metrics(config):
    frame = sample()
    frame["a"] = 100
    result = backtest_pair(frame, parameters(), config)
    assert result.metrics.total_return == result.metrics.sharpe_ratio == result.metrics.max_drawdown == 0
    assert result.metrics.number_of_trades == 0


def test_no_future_price_affects_past_signal_or_returns(config):
    original = sample()
    changed = original.copy()
    changed.iloc[4:, 0] += 100
    a = backtest_pair(original, parameters(), config)
    b = backtest_pair(changed, parameters(), config)
    pd.testing.assert_frame_equal(a.ledger.iloc[:4], b.ledger.iloc[:4])


def test_parameter_fit_must_precede_backtest(config):
    params = parameters().model_copy(update={"fitted_until": "2022-01-04"})
    with pytest.raises(DataError, match="strictly before"):
        backtest_pair(sample(), params, config)


def test_drawdown_includes_initial_capital(config):
    result = backtest_pair(sample(), parameters(), config)
    assert result.metrics.max_drawdown > 0


def test_short_borrow_cost(config):
    result = backtest_pair(sample(), parameters(), config.model_copy(update={"short_borrow_bps": 500}))
    base = backtest_pair(sample(), parameters(), config)
    assert result.metrics.total_cost > base.metrics.total_cost
