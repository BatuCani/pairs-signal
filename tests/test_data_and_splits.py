import numpy as np
from datetime import timedelta
import pandas as pd
import pytest

from pair_agent.config import ResearchConfig
from pair_agent.data.market_data import DataError, FrameMarketData, YFinanceMarketData, clean_pair, normalize
from pair_agent.research.pair_analyzer import PairAnalyzer
from pair_agent.research.splitting import chronological_split


def test_chronological_split(prices, config):
    splits = chronological_split(prices, config)
    assert splits["train"].index.max() < splits["validation"].index.min()
    assert splits["validation"].index.max() < splits["test"].index.min()
    assert sum(map(len, splits.values())) == len(prices)


def test_explicit_calendar_split(prices, config):
    cfg = ResearchConfig(start="2022-01-01", end="2026-10-19", train_end="2024-12-31", validation_end="2025-12-31")
    splits = chronological_split(prices, cfg)
    assert splits["train"].index.max().year == 2024
    assert splits["validation"].index.min().year == 2025
    assert splits["test"].index.min().year == 2026


def test_invalid_config():
    with pytest.raises(ValueError):
        ResearchConfig(train_fraction=.8, validation_fraction=.3)
    with pytest.raises(ValueError):
        ResearchConfig(train_end="2024-12-31")
    with pytest.raises(ValueError):
        ResearchConfig(transacton_cost_bps=10)


def test_no_heldout_leakage_into_fit_selection_or_score(prices, config):
    original = prices[["KO", "PEP"]]
    changed = original.copy()
    test_start = config.boundaries()[1]
    changed.loc[changed.index > test_start, "KO"] *= 2
    a = PairAnalyzer(config, FrameMarketData(original)).analyze_pair("KO", "PEP")
    b = PairAnalyzer(config, FrameMarketData(changed)).analyze_pair("KO", "PEP")
    assert a.parameters is not None
    assert a.model_dump() == b.model_dump()
    assert a.test is None


def test_future_missing_observations_do_not_move_split(prices, config):
    changed = prices.copy()
    changed.loc[changed.index > config.boundaries()[1], "KO"] = np.nan
    before = chronological_split(prices, config)["train"]
    after = chronological_split(changed, config)["train"]
    pd.testing.assert_frame_equal(before, after)


def test_selection_provider_never_receives_heldout_request(prices, config):
    class RecordingProvider(FrameMarketData):
        requests = []
        def get_adjusted_close(self, tickers, start, end):
            self.requests.append(pd.Timestamp(end))
            assert pd.Timestamp(end).date() <= config.boundaries()[1].date() + timedelta(days=1)
            return super().get_adjusted_close(tickers, start, end)
    PairAnalyzer(config, RecordingProvider(prices)).analyze_pair("KO", "PEP")


def test_missing_data_is_not_filled(prices):
    frame = prices[["KO", "PEP"]].iloc[:250].copy()
    frame.iloc[10:100, 0] = np.nan
    with pytest.raises(DataError, match="missing"):
        clean_pair(frame, 100)
    with pytest.raises(DataError, match="gap"):
        clean_pair(prices[["KO", "PEP"]].iloc[np.r_[0:100, 120:250]], 100)


def test_normalize_sort_deduplicate_and_timezone():
    frame = pd.DataFrame({"a": [1, 2, np.inf]}, index=pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-02"], utc=True))
    result = normalize(frame)
    assert result.index.is_monotonic_increasing and result.index.tz is None
    assert result.iloc[-1].isna().all()


def test_unavailable_symbol(provider):
    with pytest.raises(DataError):
        provider.get_price_history("DELISTED", "2022-01-01", "2023-01-01")


def test_yfinance_cache_avoids_second_download(tmp_path, monkeypatch):
    import yfinance as yf
    calls = []
    def download(*args, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"Adj Close": [100., 101., 102.]}, index=pd.bdate_range("2024-01-01", periods=3))
    monkeypatch.setattr(yf, "download", download)
    provider = YFinanceMarketData(tmp_path)
    a = provider.get_price_history("KO", "2024-01-01", "2024-02-01")
    b = provider.get_price_history("KO", "2024-01-01", "2024-02-01")
    pd.testing.assert_frame_equal(a, b, check_freq=False)
    assert len(calls) == 1
    assert calls[0]["auto_adjust"] is False


def test_pair_analyzer_handles_short_history(prices, config):
    result = PairAnalyzer(config, FrameMarketData(prices.iloc[:30])).analyze_pair("KO", "PEP")
    assert not result.accepted
    assert "INSUFFICIENT_DATA" in result.failure_modes
