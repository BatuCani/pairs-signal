"""Download adjusted daily closes and volumes for all pair tickers via yfinance.

Outputs data/raw/prices.csv and data/raw/volumes.csv (one column per ticker).
"""

import argparse

import yfinance as yf

from pairs_signal import config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if the output exists")
    args = parser.parse_args()

    out = config.PRICES_FILE
    if out.exists() and not args.force:
        print(f"{out} exists, skipping (use --force to refresh)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    data = yf.download(
        config.TICKERS,
        start=config.PRICE_START,
        end=config.PRICE_END,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    closes = data["Close"][config.TICKERS].dropna(how="all")
    missing = [t for t in config.TICKERS if closes[t].isna().all()]
    if missing:
        raise SystemExit(f"no price data for {missing}")
    closes.index.name = "date"
    closes.round(4).to_csv(out)
    volumes = data["Volume"][config.TICKERS].loc[closes.index]
    volumes.index.name = "date"
    volumes.to_csv(config.VOLUMES_FILE)
    print(f"Wrote {closes.shape[0]} days x {closes.shape[1]} tickers to {out}")


if __name__ == "__main__":
    main()
