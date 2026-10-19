import pytest

from pair_agent.config import ResearchConfig
from pair_agent.data.market_data import FrameMarketData
from pair_agent.data.synthetic import synthetic_prices


@pytest.fixture(scope="session")
def prices():
    return synthetic_prices()


@pytest.fixture
def config(tmp_path):
    return ResearchConfig(end="2026-10-19", data_dir=tmp_path)


@pytest.fixture
def provider(prices):
    return FrameMarketData(prices)
