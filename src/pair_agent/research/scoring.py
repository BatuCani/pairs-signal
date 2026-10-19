import math

from pair_agent.config import ResearchConfig
from pair_agent.models.enums import FailureMode as F
from pair_agent.models.schemas import PairAnalysis


def clip(value: float) -> float:
    return max(0.0, min(1.0, value)) if math.isfinite(value) else 0.0


def score_pair(pair: PairAnalysis, config: ResearchConfig) -> PairAnalysis:
    coint_ok = pair.cointegration_p_value is not None and pair.cointegration_p_value < config.p_threshold
    adf_ok = pair.adf_p_value is not None and pair.adf_p_value < config.p_threshold
    hl_ok = pair.half_life_days is not None and config.min_half_life <= pair.half_life_days <= config.max_half_life
    val, train, stress = pair.validation, pair.train, pair.validation_stressed
    stats = (float(coint_ok) + float(adf_ok) + float(hl_ok)) / 3
    oos = clip(val.sharpe_ratio / 2) if val else 0
    stability = clip(1 - max(0, train.sharpe_ratio - val.sharpe_ratio) / max(abs(train.sharpe_ratio), 1)) if val and train else 0
    dd = clip(1 - val.max_drawdown / config.max_drawdown) if val else 0
    robustness = (clip(stress.total_return / val.total_return) if val and stress and val.total_return > 0 else 0)
    activity = clip(val.number_of_trades / config.min_trades) if val else 0
    components = dict(statistical_validity=stats, out_of_sample_quality=oos, temporal_stability=stability,
                      drawdown=dd, transaction_cost_robustness=robustness, activity=activity)
    weights = dict(statistical_validity=.25, out_of_sample_quality=.25, temporal_stability=.20,
                   drawdown=.15, transaction_cost_robustness=.10, activity=.05)
    score = sum(components[k] * w for k, w in weights.items())
    failures = list(pair.failure_modes)
    if pair.correlation is None or pair.correlation < config.correlation_min:
        failures.append(F.LOW_CORRELATION)
    if not coint_ok:
        failures.append(F.HIGH_CORRELATION_NO_COINTEGRATION)
    if not adf_ok:
        failures.append(F.NON_STATIONARY_SPREAD)
    if not hl_ok:
        failures.append(F.BAD_HALF_LIFE)
    if pair.hedge_ratio is None or pair.hedge_ratio <= 0:
        failures.append(F.INVALID_HEDGE)
    if not val or val.total_return <= 0 or val.sharpe_ratio < config.min_validation_sharpe:
        failures.append(F.POOR_VALIDATION)
    if not val or val.number_of_trades < config.min_trades:
        failures.append(F.INSUFFICIENT_TRADES)
    if val and val.max_drawdown > config.max_drawdown:
        failures.append(F.EXCESSIVE_DRAWDOWN)
    if train and val and train.sharpe_ratio > 1 and val.sharpe_ratio < train.sharpe_ratio * .5:
        failures.append(F.OVERFITTING)
    if not stress or stress.total_return <= 0:
        failures.append(F.TRANSACTION_COST_SENSITIVE)
    if score < config.min_score:
        failures.append(F.LOW_SCORE)
    pair.components, pair.score = components, score
    pair.failure_modes = list(dict.fromkeys(failures))
    pair.rejection_reasons = [f.value for f in pair.failure_modes]
    pair.accepted = not pair.failure_modes
    return pair
