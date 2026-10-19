"""Fetch 8-K filings (item codes and acceptance time) for the pair firms from SEC EDGAR.

Uses the submissions API (data.sec.gov/submissions/CIK##########.json plus its older
"files" pages). acceptanceDateTime is when the filing became public, so it is what the case
builder compares with the event close to avoid lookahead. SEC asks for a descriptive
User-Agent with contact details: set SEC_USER_AGENT="name email" (default is generic).

Output: data/raw/sec_8k.csv (ticker, accepted, form, items)
"""

import argparse
import os
import time

import pandas as pd
import requests

from pairs_signal import config

CIKS = {
    "KO": 21344, "PEP": 77476, "V": 1403161, "MA": 1141391, "XOM": 34088, "CVX": 93410,
    "HD": 354950, "LOW": 60667, "GS": 886982, "MS": 895421, "UPS": 1090727, "FDX": 1048911,
    "JPM": 19617, "BAC": 70858, "WMT": 104169, "TGT": 27419, "PG": 80424, "CL": 21665,
    "CAT": 18230, "DE": 315189,
}
BASE = "https://data.sec.gov/submissions/"
SEC_FILE = config.RAW / "sec_8k.csv"

# 8-K items that signal a change in a firm's fundamentals (as opposed to 7.01 Reg FD,
# 8.01 other events, 9.01 exhibits, 5.07 shareholder votes).
FUNDAMENTAL_ITEMS = {
    "1.01": "material agreement",
    "1.02": "agreement terminated",
    "1.03": "bankruptcy",
    "2.01": "acquisition or disposal completed",
    "2.02": "results of operations",
    "2.05": "exit or restructuring costs",
    "2.06": "material impairment",
    "3.01": "delisting notice",
    "4.02": "non-reliance on prior financials",
    "5.01": "change in control",
    "5.02": "director or officer change",
}


def get_json(session: requests.Session, url: str) -> dict:
    for attempt in range(5):
        resp = session.get(url, timeout=30)
        if resp.status_code == 200:
            time.sleep(0.12)  # stay under SEC's 10 requests per second
            return resp.json()
        time.sleep(2**attempt)
    resp.raise_for_status()
    raise RuntimeError(url)


def filings_frame(block: dict) -> pd.DataFrame:
    return pd.DataFrame({k: block[k] for k in ("form", "acceptanceDateTime", "items", "filingDate")})


def fetch(ticker: str, cik: int, session: requests.Session) -> pd.DataFrame:
    root = get_json(session, f"{BASE}CIK{cik:010d}.json")
    frames = [filings_frame(root["filings"]["recent"])]
    for page in root["filings"].get("files", []):
        if page["filingTo"] < config.PRICE_START or page["filingFrom"] > config.PRICE_END:
            continue
        frames.append(filings_frame(get_json(session, BASE + page["name"])))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["form"].isin(["8-K", "8-K/A"])].copy()
    df["accepted"] = pd.to_datetime(df["acceptanceDateTime"], utc=True)
    df = df[(df["accepted"] >= pd.Timestamp(config.PRICE_START, tz="UTC")) & (df["accepted"] < pd.Timestamp(config.PRICE_END, tz="UTC"))]
    df["ticker"] = ticker
    print(f"  {ticker}: {root['name']}, {len(df)} 8-K filings")
    return df[["ticker", "accepted", "form", "items"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if SEC_FILE.exists() and not args.force:
        print(f"{SEC_FILE} exists, skipping (use --force to refresh)")
        return
    session = requests.Session()
    session.headers["User-Agent"] = os.environ.get("SEC_USER_AGENT", "pairs-signal/0.1 hackathon research")
    df = pd.concat([fetch(t, CIKS[t], session) for t in config.TICKERS]).sort_values(["ticker", "accepted"])
    SEC_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SEC_FILE, index=False)
    print(f"Wrote {len(df)} 8-K filings to {SEC_FILE}")


if __name__ == "__main__":
    main()
