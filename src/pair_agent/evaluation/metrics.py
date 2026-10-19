from statistics import mean

from pair_agent.models.schemas import AgentResult


def aggregate(results: list[AgentResult], p_threshold: float = .05) -> dict:
    if not results:
        return {"tasks": 0}
    evaluations = [r.evaluation for r in results if r.evaluation]
    selected = [p for r in results for p in r.selected_pairs]
    candidates = [p for r in results for p in r.candidates]
    return dict(tasks=len(results), mean_overall_score=mean(e.overall_score for e in evaluations) if evaluations else 0,
                valid_pair_rate=mean(bool(r.selected_pairs) and all(p.accepted for p in r.selected_pairs) for r in results),
                cointegration_pass_rate=mean(p.cointegration_p_value is not None and p.cointegration_p_value < p_threshold for p in candidates) if candidates else 0,
                mean_validation_sharpe=mean(p.validation.sharpe_ratio for p in selected if p.validation) if selected else 0,
                no_pair_decisions=sum(not r.selected_pairs for r in results),
                avg_tool_calls=mean(r.tool_call_count for r in results),
                model_errors=sum("MODEL_ERROR" in r.failure_modes for r in results))
