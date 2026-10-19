from pydantic import BaseModel, ConfigDict, Field, field_validator

from pair_agent.models.enums import FailureMode


class Schema(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class CorrelationResult(Schema):
    correlation: float
    n_observations: int


class TestResult(Schema):
    test_statistic: float | None = None
    p_value: float | None = None
    passes_threshold: bool = False
    error: str | None = None


class StationarityResult(Schema):
    adf_statistic: float | None = None
    p_value: float | None = None
    stationary: bool = False
    error: str | None = None


class SpreadParameters(Schema):
    hedge_ratio: float
    intercept: float
    mean: float
    std: float = Field(gt=0)
    fitted_until: str


class BacktestMetrics(Schema):
    total_return: float = 0
    annualized_return: float = 0
    annualized_volatility: float = 0
    sharpe_ratio: float = 0
    max_drawdown: float = 0
    number_of_trades: int = 0
    win_rate: float = 0
    total_cost: float = 0
    turnover: float = 0
    mean_abs_net_exposure: float = 0
    n_observations: int = 0


class PairAnalysis(Schema):
    ticker_a: str
    ticker_b: str
    correlation: float | None = None
    cointegration_p_value: float | None = None
    adf_p_value: float | None = None
    hedge_ratio: float | None = None
    half_life_days: float | None = None
    parameters: SpreadParameters | None = None
    train: BacktestMetrics | None = None
    validation: BacktestMetrics | None = None
    validation_stressed: BacktestMetrics | None = None
    test: BacktestMetrics | None = None
    score: float = Field(0, ge=0, le=1)
    components: dict[str, float] = Field(default_factory=dict)
    accepted: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    periods: dict[str, dict] = Field(default_factory=dict)
    data_fingerprint: str | None = None
    test_error: str | None = None

    @property
    def key(self) -> str:
        return f"{self.ticker_a}/{self.ticker_b}"


class ResearchRequest(Schema):
    request: str = "Find the best pair for KO."
    ticker: str | None = None
    tickers: list[str] = Field(default_factory=list)
    top_k: int = Field(1, ge=1, le=20)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, value):
        return value.strip().upper() if value else value

    @field_validator("tickers")
    @classmethod
    def uppercase_tickers(cls, value):
        return list(dict.fromkeys(v.strip().upper() for v in value))


class EvaluationResult(Schema):
    pair_quality: float
    statistical_validity: float
    cointegration_pass: float
    stationarity_pass: float
    out_of_sample_quality: float
    robustness: float
    tool_efficiency: float
    overall_score: float
    failure_modes: list[FailureMode] = Field(default_factory=list)


class AgentResult(Schema):
    trace_id: str
    request: ResearchRequest
    agent_version: str
    model: str | None = None
    selected_pairs: list[PairAnalysis] = Field(default_factory=list)
    candidates: list[PairAnalysis] = Field(default_factory=list)
    decision: str = "NO ROBUST PAIR FOUND"
    evaluation: EvaluationResult | None = None
    tool_call_count: int = 0
    model_call_count: int = 0
    failure_modes: list[FailureMode] = Field(default_factory=list)
