from abc import ABC, abstractmethod

from pair_agent.config import ResearchConfig
from pair_agent.data.universe import default_universe
from pair_agent.evaluation.evaluator import evaluate
from pair_agent.models.schemas import AgentResult, ResearchRequest
from pair_agent.research.candidate_selector import resolve_request
from pair_agent.research.pair_analyzer import PairAnalyzer
from pair_agent.tracing.tracer import LocalJsonlTraceSink


class ResearchAgent(ABC):
    version = "abstract"
    model = None

    def __init__(self, config: ResearchConfig, provider, universe=None, sink=None):
        self.config, self.provider = config, provider
        self.universe = universe if universe is not None else default_universe()
        self.sink = sink or LocalJsonlTraceSink(config.data_dir / "traces")

    def research(self, request: ResearchRequest) -> AgentResult:
        resolved = resolve_request(request, self.universe)
        trace_id = self.sink.start_trace(request=resolved.model_dump(mode="json"), agent_version=self.version,
                                         model=self.model, config=self.config.model_dump(mode="json"),
                                         source=getattr(self.provider, "source", type(self.provider).__name__))
        analyzer = PairAnalyzer(self.config, self.provider, self.sink)
        try:
            result = self.investigate(resolved, analyzer, trace_id)
            # Freeze the ordered selection BEFORE touching held-out data.
            for pair in result.selected_pairs:
                analyzer.evaluate_heldout(pair)
            result.tool_call_count = self.sink.tool_call_count
            result.model_call_count = self.sink.model_call_count
            result.evaluation = evaluate(result, self.config)
            result.failure_modes = result.evaluation.failure_modes
            self.sink.log_score(result.evaluation.model_dump(mode="json"))
            self.sink.end_trace(result.model_dump(mode="json"))
            return result
        except Exception as exc:
            self.sink.end_trace(dict(error=f"{type(exc).__name__}: {exc}"), status="failed")
            raise

    @abstractmethod
    def investigate(self, request, analyzer, trace_id) -> AgentResult: ...
