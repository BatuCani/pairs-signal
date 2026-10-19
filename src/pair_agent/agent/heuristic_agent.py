from pair_agent.data.market_data import DataError
from pair_agent.models.enums import FailureMode
from pair_agent.models.schemas import AgentResult, PairAnalysis
from pair_agent.research.candidate_selector import candidate_pairs
from pair_agent.tools.correlation import calculate_return_correlation

from pair_agent.agent.base import ResearchAgent


class HeuristicResearchAgent(ResearchAgent):
    version = "heuristic-v1"

    def investigate(self, request, analyzer, trace_id):
        pairs = candidate_pairs(request, self.universe)
        analyzer.run("find_peer_candidates", dict(request=request.model_dump()),
                     lambda: {"pairs": pairs, "economic_filter": "same sector; same industry preferred"})
        screened, failures = [], []
        for a, b in pairs:
            try:
                correlation = analyzer.run("screen_return_correlation", dict(ticker_a=a, ticker_b=b, scope="train"),
                                            lambda: calculate_return_correlation(analyzer.training_prices(a, b)))
                if correlation.correlation >= self.config.correlation_min:
                    screened.append((correlation.correlation, a, b))
                else:
                    failures.append(PairAnalysis(ticker_a=a, ticker_b=b, correlation=correlation.correlation,
                                                 failure_modes=[FailureMode.LOW_CORRELATION],
                                                 rejection_reasons=["return correlation below screening threshold"]))
            except DataError as exc:
                failures.append(PairAnalysis(ticker_a=a, ticker_b=b, failure_modes=[FailureMode.INSUFFICIENT_DATA],
                                             rejection_reasons=[str(exc)]))
        screened.sort(key=lambda row: (-row[0], row[1], row[2]))
        analyses = [analyzer.analyze_pair(a, b) for _, a, b in screened[:self.config.max_candidates]]
        selected = sorted((p for p in analyses if p.accepted), key=lambda p: (-p.score, p.key))[:request.top_k]
        return AgentResult(trace_id=trace_id, request=request, agent_version=self.version,
                            candidates=analyses + failures, selected_pairs=selected,
                            decision="ACCEPTED AS RESEARCH CANDIDATE" if selected else "NO ROBUST PAIR FOUND")
