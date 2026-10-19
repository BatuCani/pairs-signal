"""Fixed-share, self-financing ledger; previous-close signals, next-close fills."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pair_agent.config import ResearchConfig
from pair_agent.data.market_data import DataError
from pair_agent.models.schemas import BacktestMetrics, SpreadParameters
from pair_agent.tools.spread import construct_spread


@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    ledger: pd.DataFrame


def backtest_pair(prices: pd.DataFrame, parameters: SpreadParameters,
                  config: ResearchConfig, cost_multiplier: float = 1) -> BacktestResult:
    if prices.empty or not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        raise DataError("backtest needs positive finite prices")
    if pd.Timestamp(parameters.fitted_until) >= prices.index[0]:
        raise DataError("parameters must be fitted strictly before the backtest")
    if parameters.hedge_ratio <= 0:
        raise DataError("positive hedge ratio required for long/short shares")
    z = ((construct_spread(prices, parameters.hedge_ratio, parameters.intercept) - parameters.mean)
         / parameters.std).to_numpy()
    p = prices.to_numpy()
    equity, side = 1.0, 0
    shares = np.zeros(2)
    rows, trade_returns = [], []
    entry_equity = None
    for i in range(len(p)):
        before = equity
        held_side = side
        pnl = float(shares @ (p[i] - p[i - 1])) if i else 0.0
        equity += pnl
        short_notional = float(np.sum(np.maximum(-shares * (p[i - 1] if i else p[i]), 0)))
        borrow = short_notional * config.short_borrow_bps / 10000 / 252 if i else 0
        equity -= borrow
        cost, traded = borrow, 0.0
        signal = z[i - 1] if i else None
        desired = side
        if i:
            if side and (abs(signal) < config.exit_z or side * signal >= 0):
                desired = 0
            elif side == 0:
                desired = -1 if signal > config.entry_z else (1 if signal < -config.entry_z else 0)
        # Known evaluation horizon: liquidate at last close, charge both legs, no final entry.
        if i == len(p) - 1 or equity <= 0:
            desired = 0
        if desired != side:
            if side:
                traded = float(np.abs(shares * p[i]).sum())
                fee = traded * config.transaction_cost_bps / 10000 * cost_multiplier
                equity -= fee
                cost += fee
                trade_returns.append(equity / entry_equity - 1)
                shares[:] = 0
                entry_equity = None
            if desired and equity > 0:
                entry_equity = equity
                gross = p[i, 0] + parameters.hedge_ratio * p[i, 1]
                shares = desired * equity / gross * np.array([1.0, -parameters.hedge_ratio])
                entry_notional = float(np.abs(shares * p[i]).sum())
                traded += entry_notional
                fee = entry_notional * config.transaction_cost_bps / 10000 * cost_multiplier
                equity -= fee
                cost += fee
            side = desired
        if equity <= 0:
            raise DataError("strategy equity exhausted")
        rows.append(dict(equity=equity, daily_return=equity / before - 1, pnl=pnl,
                         cost=cost, turnover=traded / before, position=side,
                         held_position=held_side, z_score=z[i], signal_used=signal,
                         net_exposure=float((shares * p[i]).sum() / equity)))
    ledger = pd.DataFrame(rows, index=prices.index)
    r = ledger.daily_return
    volatility = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    eq = np.r_[1.0, ledger.equity.to_numpy()]
    drawdown = 1 - eq / np.maximum.accumulate(eq)
    metrics = BacktestMetrics(
        total_return=equity - 1, annualized_return=equity ** (252 / len(r)) - 1,
        annualized_volatility=volatility * np.sqrt(252),
        sharpe_ratio=float(r.mean() / volatility * np.sqrt(252)) if volatility > 1e-12 else 0,
        max_drawdown=float(drawdown.max()), number_of_trades=len(trade_returns),
        win_rate=float(np.mean(np.array(trade_returns) > 0)) if trade_returns else 0,
        total_cost=float(ledger.cost.sum()), turnover=float(ledger.turnover.sum()),
        mean_abs_net_exposure=float(ledger.net_exposure.abs().mean()), n_observations=len(r))
    return BacktestResult(metrics, ledger)
