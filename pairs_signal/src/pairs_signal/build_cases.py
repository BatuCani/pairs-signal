"""Build anonymized outlier cases (|z| >= ENTRY_Z) with outcome labels and trade returns.

An event is the first day a pair's z-score crosses |z| >= ENTRY_Z. Hedge ratio, intercept,
spread mean and spread std are frozen at the event close; the label is "reverts" if the
frozen z-score gets back inside |z| < EXIT_Z within HORIZON_DAYS trading days, else
"no_revert". The trade fades the spread at the event close and exits on reversion or at
the horizon (time stop).

Outputs:
  data/interim/cases_private.csv   real tickers and dates (never shown to a model)
  data/out/cases.jsonl             anonymized cases with split, features and outcome
"""

import json

import numpy as np
import pandas as pd

from pairs_signal import config
from pairs_signal.news import CLOSE_UTC_HOUR, anonymize, load_news
from pairs_signal.sec_events import FUNDAMENTAL_ITEMS, SEC_FILE


def split_of(day: pd.Timestamp) -> str:
    if day <= pd.Timestamp(config.TRAIN_END):
        return "train"
    if day <= pd.Timestamp(config.VAL_END):
        return "val"
    return "test"


def event_days(z: pd.Series) -> list:
    """First day |z| crosses ENTRY_Z; the next event needs |z| back below EXIT_Z first, or
    HORIZON_DAYS trading days since the last event, so one excursion is one case."""
    days, armed, last = [], True, None
    absz = z.abs().to_numpy()
    for i in range(1, len(absz)):
        if not armed and (absz[i] < config.EXIT_Z or i - last >= config.HORIZON_DAYS):
            armed = True
        if armed and absz[i] >= config.ENTRY_Z and absz[i - 1] < config.ENTRY_Z:
            days.append(z.index[i])
            armed, last = False, i
    return days


def find_events(spreads: pd.DataFrame, logp: pd.DataFrame) -> list[dict]:
    events = []
    dates = logp.index
    pos = {d: i for i, d in enumerate(dates)}
    for pair, df in spreads.groupby("pair", sort=False):
        df = df.set_index("date").sort_index()
        a, b = df["a"].iloc[0], df["b"].iloc[0]
        z = df["z"]
        for day in event_days(z):
            row = df.loc[day]
            i = pos[day]
            if i + config.HORIZON_DAYS >= len(dates):
                continue
            future = logp.iloc[i + 1 : i + 1 + config.HORIZON_DAYS]
            fut_spread = future[a] - row["alpha"] - row["beta"] * future[b]
            fut_z = (fut_spread - row["mu"]) / row["sd"]
            inside = np.flatnonzero(fut_z.abs().to_numpy() < config.EXIT_Z)
            exit_k = int(inside[0]) + 1 if len(inside) else config.HORIZON_DAYS
            exit_spread = float(fut_spread.iloc[exit_k - 1])
            # Fade the spread: short A / long beta*B when z > 0, the reverse when z < 0.
            # Spread P&L per unit of A, scaled to gross notional (1 + |beta|).
            direction = -np.sign(row["z"])
            gross = direction * (exit_spread - row["spread"]) / (1 + abs(row["beta"]))
            cost = 2 * config.COST_BPS / 1e4
            prev_z = df["z"].shift(1).loc[day]
            events.append(
                {
                    "pair": pair,
                    "a": a,
                    "b": b,
                    "date": day,
                    "exit_date": dates[i + exit_k],
                    "z": float(row["z"]),
                    "z_prev": float(prev_z),
                    "half_life": float(row["half_life"]),
                    "corr": float(row["corr"]),
                    "coint_p": float(row["coint_p"]),
                    "coint_days": int(row["coint_days"]),
                    "hedge_ratio": float(row["beta"]),
                    "ret_a": float(row["ret_a"]),
                    "ret_b": float(row["ret_b"]),
                    "label": "reverts" if len(inside) else "no_revert",
                    "holding_days": exit_k,
                    "max_abs_z_after": float(fut_z.abs().max()),
                    "gross_return": float(gross),
                    "net_return": float(gross - cost),
                }
            )
    return events


def attach_headlines(events: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Latest headline of either firm known by the event close, within NEWS_LOOKBACK_DAYS."""
    by_ticker = {t: g for t, g in news.groupby("ticker")}
    picked = []
    for ev in events.itertuples():
        best = None
        for role, ticker in (("A", ev.a), ("B", ev.b)):
            g = by_ticker.get(ticker)
            if g is None:
                continue
            window = g[(g["known_day"] <= ev.date) & (g["known_day"] > ev.date - pd.Timedelta(days=config.NEWS_LOOKBACK_DAYS))]
            if len(window) and (best is None or window["ts"].iloc[-1] > best[1]["ts"]):
                best = (role, window.iloc[-1])
        if best is None:
            picked.append({"headline_role": None, "headline_raw": None, "headline_ts": None, "headline_age_days": None})
        else:
            role, item = best
            picked.append(
                {
                    "headline_role": role,
                    "headline_raw": item["title"],
                    "headline_ts": item["ts"],
                    "headline_age_days": int((ev.date - item["known_day"]).days),
                }
            )
    return pd.concat([events.reset_index(drop=True), pd.DataFrame(picked)], axis=1)


def attach_context(cases: pd.DataFrame) -> pd.DataFrame:
    """Volume spike per leg and 8-K items accepted before the event close (no lookahead)."""
    vol = pd.read_csv(config.VOLUMES_FILE, index_col="date", parse_dates=True)
    base = vol.shift(1).rolling(config.VOLUME_DAYS).median()
    spike = vol / base
    sec = pd.read_csv(SEC_FILE, parse_dates=["accepted"])
    sec["items"] = sec["items"].fillna("")
    by_ticker = {t: g for t, g in sec.groupby("ticker")}
    rows = []
    for ev in cases.itertuples():
        close = pd.Timestamp(ev.date).tz_localize("UTC") + pd.Timedelta(hours=CLOSE_UTC_HOUR)
        row = {}
        for role, ticker in (("a", ev.a), ("b", ev.b)):
            row[f"vol_spike_{role}"] = float(spike.at[ev.date, ticker])
            g = by_ticker.get(ticker, sec.iloc[:0])
            window = g[(g["accepted"] <= close) & (g["accepted"] > close - pd.Timedelta(days=config.NEWS_LOOKBACK_DAYS))]
            items = sorted({i for s in window["items"] for i in s.split(",") if i})
            row[f"sec_items_{role}"] = ",".join(items)
        both = set(filter(None, (row["sec_items_a"] + "," + row["sec_items_b"]).split(",")))
        row["sec_fundamental"] = bool(both & FUNDAMENTAL_ITEMS.keys())
        rows.append(row)
    return pd.concat([cases.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def sec_line(items: str) -> str:
    if not items:
        return "none"
    return ", ".join(f"{i} ({FUNDAMENTAL_ITEMS[i]})" if i in FUNDAMENTAL_ITEMS else i for i in items.split(","))


def case_text(c: dict) -> str:
    rich = "expensive" if c["z"] > 0 else "cheap"
    if isinstance(c["headline"], str):
        days = int(c["headline_age_days"])
        age = "same day" if days == 0 else f"{days} day(s) before"
        headline = f'Latest headline (Company {c["headline_role"]}, {age}): "{c["headline"]}"'
    else:
        headline = f"Latest headline: none in the last {config.NEWS_LOOKBACK_DAYS} days"
    hl = "not mean reverting" if not np.isfinite(c["half_life"]) else f"{c['half_life']:.1f} trading days"
    return "\n".join(
        [
            f"Pair outlier: Company A is {rich} relative to Company B.",
            f"Spread z-score today: {c['z']:+.2f} (previous day {c['z_prev']:+.2f})",
            f"Half-life of the spread (252-day fit): {hl}",
            f"Daily return correlation (252 days): {c['corr']:.2f}",
            f"Engle-Granger p-value (252 days): {c['coint_p']:.3f}",
            f"Consecutively cointegrated for: {c['coint_days']} trading days",
            f"Hedge ratio: {c['hedge_ratio']:.2f}",
            f"Today's return: Company A {c['ret_a'] * 100:+.1f}%, Company B {c['ret_b'] * 100:+.1f}%",
            f"Volume vs {config.VOLUME_DAYS}-day median: Company A {c['vol_spike_a']:.1f}x, Company B {c['vol_spike_b']:.1f}x",
            f"SEC 8-K items filed in the last {config.NEWS_LOOKBACK_DAYS} days: Company A: {sec_line(c['sec_items_a'])}; Company B: {sec_line(c['sec_items_b'])}",
            headline,
        ]
    )


def build(include: str = "core") -> pd.DataFrame:
    summary = pd.read_csv(config.INTERIM / "pairs_summary.csv")
    allowed = {"core": ["core"], "illustrative": ["core", "illustrative"], "all": ["core", "illustrative", "excluded"]}[include]
    keep = summary[summary["status"].isin(allowed)]
    spreads = pd.read_parquet(config.INTERIM / "spreads.parquet")
    spreads = spreads[spreads["pair"].isin(keep["pair"])]
    logp = np.log(pd.read_csv(config.PRICES_FILE, index_col="date", parse_dates=True))
    events = pd.DataFrame(find_events(spreads, logp)).sort_values("date").reset_index(drop=True)
    news = load_news()
    # Without headlines the news filter cannot be judged: keep cases inside the news coverage.
    events = events[events["date"] <= news["known_day"].max()]
    cases = attach_context(attach_headlines(events, news))
    cases["split"] = cases["date"].map(split_of)
    # Embargo: drop cases whose label window reaches into the next split.
    cases = cases[cases["date"].map(split_of) == cases["exit_date"].map(split_of)].reset_index(drop=True)
    pair_ids = {p: f"P{i + 1}" for i, p in enumerate(keep.sort_values("pair")["pair"])}
    cases["pair_id"] = cases["pair"].map(pair_ids)
    cases["case_id"] = [f"c{i:05d}" for i in range(len(cases))]
    cases["headline"] = [
        anonymize(h, {a: "Company A", b: "Company B"}) if isinstance(h, str) else None
        for h, a, b in zip(cases["headline_raw"], cases["a"], cases["b"])
    ]
    return cases


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pairs",
        choices=["core", "illustrative", "all"],
        default="core",
        help="core: Engle-Granger p < PAIR_PVALUE (default); illustrative: also the not significant ones below ILLUSTRATIVE_PVALUE; all: every pair",
    )
    args = parser.parse_args()

    cases = build(include=args.pairs)
    config.OUT.mkdir(parents=True, exist_ok=True)
    cases.to_csv(config.INTERIM / "cases_private.csv", index=False)
    with (config.OUT / "cases.jsonl").open("w") as fh:
        for c in cases.to_dict("records"):
            record = {
                "case_id": c["case_id"],
                "split": c["split"],
                "pair_id": c["pair_id"],
                "input": case_text(c),
                "label": c["label"],
                "features": {
                    k: (None if isinstance(c[k], float) and not np.isfinite(c[k]) else round(c[k], 4) if isinstance(c[k], float) else c[k])
                    for k in ("z", "z_prev", "half_life", "corr", "coint_p", "coint_days", "hedge_ratio", "ret_a", "ret_b",
                              "vol_spike_a", "vol_spike_b", "sec_items_a", "sec_items_b", "sec_fundamental", "headline_age_days")
                },
                "headline": c["headline"] if isinstance(c["headline"], str) else None,
                "headline_role": c["headline_role"] if isinstance(c["headline_role"], str) else None,
                "outcome": {
                    "holding_days": c["holding_days"],
                    "max_abs_z_after": round(c["max_abs_z_after"], 3),
                    "gross_return": round(c["gross_return"], 6),
                    "net_return": round(c["net_return"], 6),
                },
            }
            fh.write(json.dumps(record) + "\n")
    stats = cases.groupby("split")["label"].value_counts().unstack(fill_value=0)
    stats["revert_rate"] = (stats["reverts"] / stats.sum(axis=1)).round(3)
    print(stats.to_string())
    print(f"\n{len(cases)} cases, {cases['headline'].notna().mean():.0%} with a headline -> {config.OUT / 'cases.jsonl'}")


if __name__ == "__main__":
    main()
