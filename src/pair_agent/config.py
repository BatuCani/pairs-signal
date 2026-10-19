from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    start: date = date(2022, 1, 1)
    end: date = Field(default_factory=lambda: date.today() + timedelta(days=1))  # exclusive
    train_end: date | None = None
    validation_end: date | None = None
    train_fraction: float = Field(0.60, gt=0, lt=1)
    validation_fraction: float = Field(0.20, gt=0, lt=1)
    min_train: int = Field(126, ge=60)
    min_eval: int = Field(40, ge=10)
    max_missing_fraction: float = Field(0.05, ge=0, lt=1)
    max_gap_business_days: int = Field(5, ge=1)
    p_threshold: float = Field(0.05, gt=0, lt=1)
    correlation_min: float = Field(0.5, ge=0, le=1)
    min_half_life: float = Field(1.0, gt=0)
    max_half_life: float = Field(60, gt=0)
    entry_z: float = Field(2.0, gt=0)
    exit_z: float = Field(0.5, ge=0)
    transaction_cost_bps: float = Field(10, ge=0)
    short_borrow_bps: float = Field(0, ge=0)
    min_validation_sharpe: float = 0.25
    min_trades: int = Field(3, ge=1)
    max_drawdown: float = Field(0.2, gt=0, le=1)
    min_score: float = Field(0.55, ge=0, le=1)
    max_candidates: int = Field(12, ge=1, le=100)
    max_model_steps: int = Field(20, ge=1, le=100)
    data_dir: Path = Path("data")

    @model_validator(mode="after")
    def check(self):
        if self.start >= self.end:
            raise ValueError("start must precede exclusive end")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("leave a nonempty held-out fraction")
        if (self.train_end is None) != (self.validation_end is None):
            raise ValueError("set both train_end and validation_end or neither")
        if self.train_end and not self.start < self.train_end < self.validation_end < self.end:
            raise ValueError("require start < train_end < validation_end < end")
        if self.exit_z >= self.entry_z or self.min_half_life > self.max_half_life:
            raise ValueError("invalid thresholds")
        return self

    def boundaries(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        if self.train_end:
            return pd.Timestamp(self.train_end), pd.Timestamp(self.validation_end)
        # Calendar boundaries are fixed BEFORE inspecting any prices or missing data.
        calendar = pd.bdate_range(self.start, self.end, inclusive="left")
        if len(calendar) < 3:
            raise ValueError("date range too short")
        a = int(len(calendar) * self.train_fraction) - 1
        b = int(len(calendar) * (self.train_fraction + self.validation_fraction)) - 1
        return calendar[max(0, a)], calendar[max(1, b)]
