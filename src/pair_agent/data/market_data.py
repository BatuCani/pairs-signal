"""Replaceable adjusted-price providers. Never fill prices across missing periods."""
from abc import ABC, abstractmethod
from hashlib import sha256
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd


class DataError(ValueError):
    pass


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None).normalize()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return frame


def clean_pair(frame: pd.DataFrame, minimum: int, max_missing: float = 0.05, max_gap: int = 5):
    if frame.empty or frame.shape[1] != 2:
        raise DataError("two nonempty price series are required")
    values = frame.where(frame > 0)
    missing = 1 - len(values.dropna()) / len(values)
    if missing > max_missing:
        raise DataError(f"missing/invalid aligned observations: {missing:.1%}")
    values = values.dropna()
    if len(values) < minimum:
        raise DataError(f"need {minimum} aligned observations; have {len(values)}")
    if len(values) > 1:
        gaps = np.busday_count(values.index[:-1].values.astype('datetime64[D]'),
                               values.index[1:].values.astype('datetime64[D]'))
        if max(gaps) > max_gap:
            raise DataError("large gap in price history")
    if (values.std() < 1e-10).any():
        raise DataError("constant price history")
    return values


class MarketDataProvider(ABC):
    @abstractmethod
    def get_price_history(self, ticker: str, start: str, end: str) -> pd.DataFrame: ...

    def get_adjusted_close(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        frames, errors = [], []
        for ticker in dict.fromkeys(tickers):
            try:
                frames.append(self.get_price_history(ticker, start, end)["Adj Close"].rename(ticker))
            except DataError as exc:
                errors.append(f"{ticker}: {exc}")
        if errors:
            warnings.warn("Unavailable symbols: " + "; ".join(errors), stacklevel=2)
        if not frames:
            raise DataError("no available price history")
        return normalize(pd.concat(frames, axis=1))


class YFinanceMarketData(MarketDataProvider):
    def __init__(self, cache_dir: Path = Path("data/raw"), refresh: bool = False):
        self.cache_dir, self.refresh = Path(cache_dir), refresh

    def get_price_history(self, ticker, start, end):
        import yfinance as yf
        ticker = ticker.upper()
        if not re.fullmatch(r"[A-Z0-9.^=-]{1,20}", ticker):
            raise DataError("invalid ticker")
        key = sha256(f"yf-adjusted-v1|{ticker}|{start}|{end}".encode()).hexdigest()[:20]
        path = self.cache_dir / f"{ticker}-{key}.csv"
        if path.exists() and not self.refresh:
            try:
                frame = normalize(pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip"))
            except (ValueError, OSError) as exc:
                raise DataError(f"invalid cache {path}; use --refresh") from exc
        else:
            yf.set_tz_cache_location(str(self.cache_dir / "yfinance"))
            try:
                frame = yf.download(ticker, start=start, end=end, auto_adjust=False,
                                    progress=False, threads=False, multi_level_index=False,
                                    timeout=20)
            except Exception as exc:
                raise DataError(f"download failed ({type(exc).__name__})") from exc
            if frame is None or frame.empty:
                raise DataError("empty or unavailable symbol")
            frame = normalize(frame)
            if "Adj Close" not in frame:
                raise DataError("provider did not return adjusted close")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Single-writer MVP cache. Refresh explicitly when data revisions are desired.
            temp = path.with_suffix(".tmp")
            frame.to_csv(temp)
            temp.replace(path)
        if "Adj Close" not in frame or frame["Adj Close"].dropna().empty:
            raise DataError("no adjusted prices")
        return frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end))]


class FrameMarketData(MarketDataProvider):
    """Local/synthetic prices; injected into exactly the same research path."""
    def __init__(self, prices: pd.DataFrame, source: str = "FrameMarketData"):
        self.prices = normalize(prices)
        self.source = source

    def get_price_history(self, ticker, start, end):
        if ticker not in self.prices:
            raise DataError(f"unknown symbol {ticker}")
        series = self.prices.loc[(self.prices.index >= pd.Timestamp(start)) &
                                 (self.prices.index < pd.Timestamp(end)), ticker]
        if series.dropna().empty:
            raise DataError("no observations in requested period")
        return series.to_frame("Adj Close")


def get_price_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    return YFinanceMarketData().get_price_history(ticker, start, end)


def get_adjusted_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    return YFinanceMarketData().get_adjusted_close(tickers, start, end)
