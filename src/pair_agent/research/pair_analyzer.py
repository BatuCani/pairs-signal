from hashlib import sha256
from datetime import timedelta
from time import perf_counter

import numpy as np
import pandas as pd

from pair_agent.config import ResearchConfig
from pair_agent.data.market_data import DataError, MarketDataProvider, YFinanceMarketData, clean_pair
from pair_agent.models.enums import FailureMode
from pair_agent.models.schemas import PairAnalysis
from pair_agent.research.scoring import score_pair
from pair_agent.research.splitting import chronological_split
from pair_agent.tools.backtest import backtest_pair
from pair_agent.tools.cointegration import calculate_cointegration
from pair_agent.tools.correlation import calculate_return_correlation
from pair_agent.tools.half_life import estimate_half_life
from pair_agent.tools.spread import construct_spread, estimate_spread_parameters
from pair_agent.tools.stationarity import calculate_stationarity


class PairAnalyzer:
    def __init__(self, config: ResearchConfig, provider: MarketDataProvider, sink=None):
        self.config, self.provider, self.sink = config, provider, sink
        self.cache: dict[tuple[str, str], PairAnalysis] = {}

    def run(self, name, inputs, operation):
        started = perf_counter()
        try:
            value = operation()
        except Exception as exc:
            if self.sink:
                self.sink.log_tool_call(name, inputs, {}, (perf_counter() - started) * 1000,
                                        f"{type(exc).__name__}: {exc}")
            raise
        if self.sink:
            if hasattr(value, "model_dump"):
                output = value.model_dump(mode="json")
            elif hasattr(value, "metrics"):
                output = value.metrics.model_dump(mode="json")
            elif isinstance(value, pd.DataFrame):
                output = dict(rows=len(value), columns=list(value.columns))
            elif isinstance(value, dict):
                output = value
            else:
                output = {"value": value}
            self.sink.log_tool_call(name, inputs, output, (perf_counter() - started) * 1000)
        return value

    def load(self, a, b, include_test=False):
        # Selection tools physically request only train+validation; test isn't loaded.
        end = str(self.config.end) if include_test else str(self.config.boundaries()[1].date() + timedelta(days=1))
        frame = self.provider.get_adjusted_close([a, b], str(self.config.start), end)
        if a not in frame or b not in frame:
            raise DataError("one or both symbols unavailable")
        return frame[[a, b]]

    def training_prices(self, a, b):
        # Correlation ranking downloads only the training window.
        end = str(self.config.boundaries()[0].date() + timedelta(days=1))
        frame = self.provider.get_adjusted_close([a, b], str(self.config.start), end)
        if a not in frame or b not in frame:
            raise DataError("symbol unavailable")
        train = chronological_split(frame[[a, b]], self.config)["train"]
        return self.clean(train, self.config.min_train)

    def clean(self, frame, minimum):
        return clean_pair(frame, minimum, self.config.max_missing_fraction, self.config.max_gap_business_days)

    def analyze_pair(self, a: str, b: str) -> PairAnalysis:
        if (a, b) in self.cache:
            return self.cache[a, b]
        pair = PairAnalysis(ticker_a=a, ticker_b=b)
        try:
            raw = self.run("load_adjusted_prices", dict(ticker_a=a, ticker_b=b, scope="train_validation"),
                           lambda: self.load(a, b))
            splits = chronological_split(raw, self.config)
            train = self.clean(splits["train"], self.config.min_train)
            validation = self.clean(splits["validation"], self.config.min_eval)
            for label, frame in [("train", train), ("validation", validation)]:
                pair.periods[label] = dict(start=frame.index[0].isoformat(),
                                          end=frame.index[-1].isoformat(), n_observations=len(frame))
            pair.data_fingerprint = sha256(pd.util.hash_pandas_object(raw, index=True).values.tobytes()).hexdigest()
            inputs = dict(ticker_a=a, ticker_b=b, scope="train", **pair.periods["train"])
            pair.correlation = self.run("calculate_correlation", inputs,
                                        lambda: calculate_return_correlation(train)).correlation
            cointegration = self.run("calculate_cointegration", inputs,
                                      lambda: calculate_cointegration(train, self.config.p_threshold))
            pair.cointegration_p_value = cointegration.p_value
            parameters = self.run("estimate_spread_parameters", inputs, lambda: estimate_spread_parameters(train))
            pair.parameters, pair.hedge_ratio = parameters, parameters.hedge_ratio
            spread = construct_spread(train, parameters.hedge_ratio, parameters.intercept)
            pair.adf_p_value = self.run("calculate_stationarity", inputs,
                                        lambda: calculate_stationarity(spread, self.config.p_threshold)).p_value
            pair.half_life_days = self.run("estimate_half_life", inputs, lambda: estimate_half_life(spread))
            if parameters.hedge_ratio <= 0:
                pair.failure_modes.append(FailureMode.INVALID_HEDGE)
            else:
                # Causal train diagnostic: fit on first 60%, replay remaining 40%.
                cut = max(30, int(len(train) * .6))
                inner_parameters = estimate_spread_parameters(train.iloc[:cut])
                pair.periods["train_backtest"] = dict(start=train.index[cut].isoformat(),
                                                       end=train.index[-1].isoformat(), n_observations=len(train)-cut)
                pair.train = self.run("backtest_train", dict(parameters=inner_parameters.model_dump(),
                                      scope="train_tail"),
                                      lambda: backtest_pair(train.iloc[cut:], inner_parameters, self.config)).metrics
                for label, multiplier in [("validation", 1), ("validation_stressed", 2)]:
                    metrics = self.run("backtest_pair", dict(scope=label, cost_multiplier=multiplier,
                                        parameters=parameters.model_dump(), **pair.periods["validation"]),
                                        lambda m=multiplier: backtest_pair(validation, parameters, self.config, m)).metrics
                    setattr(pair, label, metrics)
        except (DataError, ValueError, np.linalg.LinAlgError) as exc:
            pair.failure_modes.append(FailureMode.INSUFFICIENT_DATA)
            pair.rejection_reasons.append(str(exc))
        detail = pair.rejection_reasons[:]
        pair = score_pair(pair, self.config)
        pair.rejection_reasons.extend(detail)
        self.cache[a, b] = pair
        return pair

    def evaluate_heldout(self, pair: PairAnalysis) -> PairAnalysis:
        # Call ONLY after freezing selection. Never updates score or acceptance.
        if not pair.accepted or not pair.parameters:
            return pair
        try:
            test = self.clean(chronological_split(self.load(pair.ticker_a, pair.ticker_b, True), self.config)["test"],
                              self.config.min_eval)
            pair.test = self.run("heldout_backtest", dict(pair=pair.key, scope="heldout_after_selection"),
                                  lambda: backtest_pair(test, pair.parameters, self.config)).metrics
            pair.periods["test"] = dict(start=test.index[0].isoformat(), end=test.index[-1].isoformat(),
                                        n_observations=len(test))
        except (DataError, ValueError) as exc:
            pair.test_error = str(exc)
        return pair


def analyze_pair(ticker_a: str, ticker_b: str, config: ResearchConfig,
                 provider: MarketDataProvider | None = None) -> PairAnalysis:
    analyzer = PairAnalyzer(config, provider or YFinanceMarketData(config.data_dir / "raw"))
    return analyzer.evaluate_heldout(analyzer.analyze_pair(ticker_a, ticker_b))
