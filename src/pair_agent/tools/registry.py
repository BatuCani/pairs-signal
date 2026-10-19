"""Allowlisted JSON tool boundary shared by the LLM and future integrations."""
from pydantic import Field

from pair_agent.models.schemas import Schema
from pair_agent.research.candidate_selector import candidate_pairs, find_peer_candidates
from pair_agent.tools.cointegration import calculate_cointegration
from pair_agent.tools.correlation import calculate_return_correlation
from pair_agent.tools.half_life import estimate_half_life
from pair_agent.tools.spread import construct_spread, estimate_spread_parameters
from pair_agent.tools.stationarity import calculate_stationarity


class PairArguments(Schema):
    ticker_a: str = Field(min_length=1, max_length=20)
    ticker_b: str = Field(min_length=1, max_length=20)


class PeerArguments(Schema):
    ticker: str


class FinishArguments(Schema):
    pairs: list[str] = Field(max_length=20)


TOOL_DESCRIPTIONS = {
    "find_peer_candidates": "Find economically related peers within the allowed request scope.",
    "calculate_correlation": "Training return correlation; screening evidence only.",
    "calculate_cointegration": "Engle-Granger test on training prices only.",
    "calculate_stationarity": "ADF test on training OLS residual spread.",
    "estimate_half_life": "Discrete AR(1) half-life on training spread; null if invalid.",
    "analyze_pair": "Full deterministic training/validation analysis with acceptance gates; no held-out data.",
    "backtest_pair": "Full analysis including causal train diagnostic, validation, doubled-cost stress.",
    "finish_research": "Freeze ordered selection of previously analyzed accepted pair keys, or [].",
}


class ToolRegistry:
    def __init__(self, analyzer, request, universe):
        self.analyzer, self.request, self.universe = analyzer, request, universe
        self.allowed = candidate_pairs(request, universe)

    @staticmethod
    def schemas():
        result = []
        for name, description in TOOL_DESCRIPTIONS.items():
            args = FinishArguments if name == "finish_research" else (PeerArguments if name == "find_peer_candidates" else PairArguments)
            result.append(dict(type="function", function=dict(name=name, description=description,
                                                               parameters=args.model_json_schema())))
        return result

    def execute(self, name: str, arguments: dict) -> dict:
        if name not in TOOL_DESCRIPTIONS or name == "finish_research":
            raise ValueError("unknown or non-executable tool")
        if name == "find_peer_candidates":
            args = PeerArguments.model_validate(arguments)
            allowed_symbols = {t for p in self.allowed for t in p}
            if args.ticker not in allowed_symbols:
                raise ValueError("ticker outside request scope")
            return self.analyzer.run(name, arguments, lambda: {"peers": [t for t in find_peer_candidates(args.ticker, self.universe)
                                                                                        if (args.ticker, t) in self.allowed or (t, args.ticker) in self.allowed]})
        args = PairArguments.model_validate(arguments)
        a, b = args.ticker_a, args.ticker_b
        if (a, b) not in self.allowed:
            raise ValueError("pair outside eligible request scope or orientation")
        if name in {"analyze_pair", "backtest_pair"}:
            if (a, b) not in self.analyzer.cache and len(self.analyzer.cache) >= self.analyzer.config.max_candidates:
                raise ValueError("deep analysis budget exhausted")
            return self.analyzer.run(name, arguments, lambda: self.analyzer.analyze_pair(a, b)).model_dump(mode="json")
        def calculate():
            train = self.analyzer.training_prices(a, b)
            if name == "calculate_correlation":
                return calculate_return_correlation(train)
            if name == "calculate_cointegration":
                return calculate_cointegration(train, self.analyzer.config.p_threshold)
            parameters = estimate_spread_parameters(train)
            spread = construct_spread(train, parameters.hedge_ratio, parameters.intercept)
            if name == "calculate_stationarity":
                return calculate_stationarity(spread, self.analyzer.config.p_threshold)
            return {"half_life_days": estimate_half_life(spread)}
        result = self.analyzer.run(name, dict(**arguments, scope="train"), calculate)
        return result.model_dump(mode="json") if hasattr(result, "model_dump") else result
