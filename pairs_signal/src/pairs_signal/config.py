"""Shared paths, universe and parameters."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
OUT = DATA / "out"
TASKS = ROOT / "tasks"

# Classic same-sector pairs. Cointegration filtering decides which survive.
PAIRS: list[tuple[str, str]] = [
    ("KO", "PEP"),
    ("V", "MA"),
    ("XOM", "CVX"),
    ("HD", "LOW"),
    ("GS", "MS"),
    ("UPS", "FDX"),
    ("JPM", "BAC"),
    ("WMT", "TGT"),
    ("PG", "CL"),
    ("CAT", "DE"),
]
TICKERS: list[str] = sorted({t for pair in PAIRS for t in pair})

# FNSPID news ends in late 2023; prices cover the same window.
PRICE_START = "2010-01-01"
PRICE_END = "2024-01-01"

# All_external.csv (5.7 GB) covers these tickers until 2020-06-11 (no GS, no CVX);
# nasdaq_exteral_data.csv (23 GB, ~1 h to stream) reaches 2023: fetch-news --source nasdaq.
FNSPID_SOURCES = {
    "external": "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv",
    "nasdaq": "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv",
}
NEWS_FILE = RAW / "fnspid_news.csv"
PRICES_FILE = RAW / "prices.csv"
VOLUMES_FILE = RAW / "volumes.csv"

# Pair analysis
FORMATION_DAYS = 252  # rolling window for hedge ratio and cointegration test
ZSCORE_DAYS = 60  # rolling window for spread mean/std
COINT_PVALUE = 0.05  # significance level for the rolling (252-day) Engle-Granger test
# Pair filter on the full-sample Engle-Granger p-value: below PAIR_PVALUE a pair is "core"
# (GS/MS, CAT/DE, KO/PEP over 2010-2023); below ILLUSTRATIVE_PVALUE it is "illustrative"
# (not significant, only used with build_cases --pairs illustrative).
PAIR_PVALUE = 0.05
ILLUSTRATIVE_PVALUE = 0.15
ENTRY_Z = 2.0
EXIT_Z = 0.5
HORIZON_DAYS = 10  # trading days the spread has to revert in
COST_BPS = 5.0  # one-way cost per traded notional (bps); a trade pays it on entry and exit

# Headline lookup: only news strictly before the event day's close, at most this old.
NEWS_LOOKBACK_DAYS = 7
VOLUME_DAYS = 20  # volume spike = event-day volume / median of the previous VOLUME_DAYS days

# Time-based splits on event / headline date (test = most recent period, never trained on).
# Chosen inside the news coverage (All_external ends 2020-06-11): test is the last 18 months.
TRAIN_END = "2017-12-31"
VAL_END = "2018-12-31"

TEACHER_MODEL = "claude-haiku-5-5"

# Name aliases per ticker, replaced by "Company A"/"Company B" when anonymizing text.
COMPANY_ALIASES: dict[str, list[str]] = {
    "KO": ["Coca-Cola", "Coca Cola", "COCA-COLA", "Coke"],
    "PEP": ["PepsiCo", "Pepsico", "PEPSICO", "Pepsi"],
    "V": ["Visa"],
    "MA": ["Mastercard", "MasterCard"],
    "XOM": ["Exxon Mobil", "ExxonMobil", "Exxon"],
    "CVX": ["Chevron"],
    "HD": ["Home Depot"],
    "LOW": ["Lowe's", "Lowe’s", "Lowes"],
    "GS": ["Goldman Sachs", "Goldman"],
    "MS": ["Morgan Stanley"],
    "UPS": ["United Parcel Service", "UPS"],
    "FDX": ["FedEx", "Federal Express"],
    "JPM": ["JPMorgan Chase", "JP Morgan", "JPMorgan", "J.P. Morgan"],
    "BAC": ["Bank of America", "BofA", "BoA"],
    "WMT": ["Walmart", "Wal-Mart", "Wal Mart"],
    "TGT": ["Target"],
    "PG": ["Procter & Gamble", "Procter and Gamble", "P&G"],
    "CL": ["Colgate-Palmolive", "Colgate Palmolive", "Colgate"],
    "CAT": ["Caterpillar"],
    "DE": ["Deere & Company", "Deere & Co", "John Deere", "Deere"],
}
