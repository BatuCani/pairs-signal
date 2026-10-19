import json
from pathlib import Path

from pair_agent.models.schemas import Schema


class Stock(Schema):
    ticker: str
    sector: str
    industry: str


GROUPS = {
    ("Technology", "Semiconductors"): "NVDA AMD INTC QCOM AVGO MU TXN ADI",
    ("Technology", "Software"): "MSFT ORCL CRM ADBE",
    ("Technology", "Consumer Technology"): "AAPL GOOGL META AMZN",
    ("Financials", "Banks"): "JPM BAC C WFC GS MS",
    ("Energy", "Integrated Oil"): "XOM CVX COP EOG SLB OXY",
    ("Consumer Staples", "Beverages"): "KO PEP KDP MNST",
    ("Consumer Staples", "Household Products"): "PG CL KMB CHD",
    ("Healthcare", "Pharmaceuticals"): "JNJ PFE MRK ABBV LLY BMY",
    ("Healthcare", "Managed Care"): "UNH CI CVS HUM",
}


def default_universe() -> list[Stock]:
    # Curated convenience metadata, not a point-in-time security master.
    return [Stock(ticker=t, sector=s, industry=i) for (s, i), ts in GROUPS.items() for t in ts.split()]


def load_universe(path: Path | None = None) -> list[Stock]:
    stocks = default_universe() if path is None else [Stock.model_validate(x) for x in json.loads(path.read_text())]
    if len({s.ticker for s in stocks}) != len(stocks):
        raise ValueError("duplicate tickers in universe")
    return stocks
