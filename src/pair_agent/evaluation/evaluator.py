from statistics import mean

from pair_agent.config import ResearchConfig
from pair_agent.evaluation.failures import classify_failures
from pair_agent.models.enums import FailureMode as F
from pair_agent.models.schemas import AgentResult, EvaluationResult


def evaluate(result: AgentResult, config: ResearchConfig) -> EvaluationResult:
    failures = classify_failures(result)
    pairs = result.selected_pairs
    if pairs:
        quality = mean(p.score for p in pairs)
        coint = mean(float(p.cointegration_p_value is not None and p.cointegration_p_value < config.p_threshold) for p in pairs)
        stationary = mean(float(p.adf_p_value is not None and p.adf_p_value < config.p_threshold) for p in pairs)
        statistics = mean(p.components.get("statistical_validity", 0) for p in pairs)
        oos = mean(p.components.get("out_of_sample_quality", 0) for p in pairs)
        robustness = mean(p.components.get("transaction_cost_robustness", 0) for p in pairs)
    else:
        quality = coint = stationary = statistics = oos = robustness = 0.0
    efficiency = min(1.0, max(1, len(result.candidates)) * 12 / max(1, result.tool_call_count))
    if pairs:
        overall = .4 * quality + .2 * statistics + .2 * oos + .1 * robustness + .1 * efficiency
        if F.FORCED_PAIR_SELECTION in failures:
            overall = 0
    else:
        # Rejection is permissible but is not proof of robustness or a perfect outcome.
        has_evidence = any(p.parameters is not None for p in result.candidates)
        valid_available = any(p.accepted for p in result.candidates)
        overall = .5 * efficiency if has_evidence and not valid_available else 0
    return EvaluationResult(pair_quality=quality, statistical_validity=statistics,
                            cointegration_pass=coint, stationarity_pass=stationary,
                            out_of_sample_quality=oos, robustness=robustness,
                            tool_efficiency=efficiency, overall_score=overall, failure_modes=failures)
