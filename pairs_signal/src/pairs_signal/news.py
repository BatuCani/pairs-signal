"""Loading FNSPID headlines and anonymizing company names and dates in text."""

import re
from functools import cache

import pandas as pd

from pairs_signal import config

# US close is 16:00 ET = 20:00 UTC in summer (21:00 in winter). Headlines stamped before
# 20:00 UTC on the event day count as known at the close; later ones go to the next day.
CLOSE_UTC_HOUR = 20

YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
MONTH_DAY_RE = re.compile(
    r"\b(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|June?|July?|Aug(ust)?|"
    r"Sep(t(ember)?)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)\.?\s+\d{1,2}(st|nd|rd|th)?\b"
)


def load_news() -> pd.DataFrame:
    """All filtered headlines with a UTC timestamp and the trading day they are known by."""
    news = pd.read_csv(config.NEWS_FILE, dtype=str, keep_default_na=False)
    news = news[news["title"].str.strip() != ""].copy()
    news["ts"] = pd.to_datetime(news["date"].str.replace(" UTC", "", regex=False), utc=True, errors="coerce")
    news = news.dropna(subset=["ts"])
    # Calendar day on which the headline is public before the close.
    after_close = news["ts"].dt.hour >= CLOSE_UTC_HOUR
    news["known_day"] = news["ts"].dt.tz_localize(None).dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")
    news = news.drop_duplicates(subset=["ticker", "title"]).sort_values("ts")
    return news[["ts", "known_day", "ticker", "title", "publisher", "url"]].reset_index(drop=True)


@cache
def _alias_pattern(ticker: str) -> re.Pattern:
    names = sorted(config.COMPANY_ALIASES.get(ticker, []), key=len, reverse=True)
    # "Target" is also a word in nearly every analyst headline ("Raises Price Target").
    guard = r"(?<!Price )(?<!price )" if ticker == "TGT" else ""
    plural = "" if ticker == "TGT" else "s?"  # "Morgan Stanleys" (apostrophe dropped)
    parts = [rf"{guard}\b{re.escape(n)}{plural}(?=\W|$)" for n in names]
    # Bare tickers only when they are not also common words or single letters...
    if len(ticker) >= 2 and ticker not in {"LOW", "MA", "MS", "DE", "CL", "HD", "GS", "PG"}:
        parts.append(rf"\b{re.escape(ticker)}\b")
    # ...but always inside ticker lists such as "(GLD, USO, GS)" or "(V, MA)".
    parts.append(rf"(?:(?<=\()|(?<=,)|(?<=, )){re.escape(ticker)}(?=\s*[,)])")
    return re.compile("|".join(parts))


@cache
def _exchange_tag(ticker: str) -> re.Pattern:
    """Exchange tags such as "(NYSE:KO)" or "NASDAQ: PEP"; they only repeat the name."""
    return re.compile(rf"\s*\(?\b(?:NYSE|NASDAQ|Nasdaq)\s*:\s*{re.escape(ticker)}\b\)?")


def anonymize(text: str, mapping: dict[str, str]) -> str:
    """Replace company names (ticker -> placeholder) and explicit dates/years."""
    for ticker, placeholder in mapping.items():
        text = _exchange_tag(ticker).sub("", text)
        text = _alias_pattern(ticker).sub(placeholder, text)
    text = re.sub(r"(Company [AB])(?:,? Inc\.?)? \(\1\)", r"\1", text)  # "Company A (Company A)"
    text = MONTH_DAY_RE.sub("[date]", text)
    text = YEAR_RE.sub("[year]", text)
    return re.sub(r"\s{2,}", " ", text).strip()
