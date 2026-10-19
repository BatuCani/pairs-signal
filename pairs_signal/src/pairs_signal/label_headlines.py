"""Headline classification dataset: anonymize pair-firm headlines, split by time, label.

Labels come from the teacher (Claude, ANTHROPIC_API_KEY from the environment). With
--dry-run no API call is made and a keyword rule labeler stands in, so the rest of the
pipeline runs end to end; those rows carry label_source="rules".

Teacher answers are cached in data/interim/teacher_labels.jsonl by headline text, so a
re-run only pays for new headlines.

Outputs:
  data/interim/headlines_private.csv   real tickers and timestamps
  data/out/headlines.jsonl             anonymized headlines with split and label
  data/out/headline_review.csv         50 stratified rows for manual label checking
"""

import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from pairs_signal import config
from pairs_signal.build_cases import split_of
from pairs_signal.news import anonymize, load_news

LABELS_FILE = config.TASKS / "headline_labels.json"
CACHE_FILE = config.INTERIM / "teacher_labels.jsonl"
BATCH_SIZE = 25
REVIEW_SIZE = 50

# Keyword stand-in for the teacher, checked in order; first match wins.
RULES: list[tuple[str, str]] = [
    ("management_change", r"\b(CEO|CFO|COO|chief executive|chairman|chairwoman|board|executive|president)\b.*\b(steps? down|resign|retire|appoint|named|names|depart|succe|replace|hire|exit|ousted|fired)|\b(new|names|appoints) (CEO|CFO|chief)"),
    ("m_and_a", r"\b(acquir|acquisition|merger|merges?\b|takeover|buyout|agrees? to (buy|sell)|divest|spin[- ]?off|spinoff|joint venture|stake in|bid for)"),
    ("legal_regulatory", r"\b(lawsuit|sued|sues|settle|fine[ds]?\b|penalt|probe|investigat|SEC\b|DOJ|FTC|antitrust|court|judge|regulator|recall|strike|union|subpoena|class action|charges)"),
    ("analyst_rating", r"\b(upgrade|downgrade|price target|initiat|reiterat|maintains|outperform|underperform|overweight|underweight|buy rating|sell rating|neutral rating|analyst)"),
    ("earnings_warning", r"\b(miss(es|ed)?|profit warning|warns|cuts? (guidance|outlook|forecast|dividend)|lowers? (guidance|outlook|forecast)|weak|disappoint|falls short|write[- ]?down|impairment|loss widens|slump)"),
    ("guidance_up", r"\b(beats?|tops|raises? (guidance|outlook|forecast|dividend)|record (profit|revenue|sales|quarter)|dividend (hike|increase)|buyback|repurchase|surpass)"),
    ("macro_sector", r"\b(Fed\b|inflation|interest rate|rates|economy|recession|GDP|jobs report|oil price|crude|tariff|trade war|sector|industry|banks|retailers|oil stocks|consumer spending|treasury)"),
]


def load_task() -> dict:
    return json.loads(LABELS_FILE.read_text())


def rule_label(text: str) -> str:
    for label, pattern in RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return "noise"


def build_headlines() -> pd.DataFrame:
    news = load_news()
    roles = {}
    for a, b in config.PAIRS:
        roles[a] = ("A", {a: "Company A", b: "Company B"})
        roles[b] = ("B", {a: "Company A", b: "Company B"})
    news = news[news["ticker"].isin(roles)].copy()
    news["role"] = news["ticker"].map(lambda t: roles[t][0])
    news["pair"] = news["ticker"].map(lambda t: "/".join(k for k in roles[t][1]))
    news["text"] = [anonymize(t, roles[tk][1]) for t, tk in zip(news["title"], news["ticker"])]
    news = news[news["text"].str.len() >= 15]
    # One row per anonymized text, earliest occurrence, so no headline sits in two splits.
    news = news.sort_values("ts").drop_duplicates("text").reset_index(drop=True)
    news["split"] = news["known_day"].map(split_of)
    news["headline_id"] = [f"h{i:06d}" for i in range(len(news))]
    return news


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def load_cache() -> dict[str, str]:
    if not CACHE_FILE.exists():
        return {}
    cache = {}
    for line in CACHE_FILE.read_text().splitlines():
        row = json.loads(line)
        cache[row["key"]] = row["label"]
    return cache


def teacher_label(texts: list[str], model: str, workers: int) -> dict[str, str]:
    """Label texts with the teacher in batches; returns text_key -> label (cached on disk)."""
    from pairs_signal.llm import Claude

    task = load_task()
    labels = list(task["labels"])
    system = (
        task["task_description"].replace("Answer with the label only.", "")
        + "\n\nLabels:\n"
        + "\n".join(f"- {k}: {v}" for k, v in task["labels"].items())
        + "\n\nYou get a JSON list of headlines with ids. Return one label per id."
    )
    schema = {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}, "label": {"type": "string", "enum": labels}},
                    "required": ["id", "label"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }
    cache = load_cache()
    todo = sorted({t for t in texts if text_key(t) not in cache})
    print(f"teacher: {len(texts) - len(todo)} cached, {len(todo)} to label with {model}")
    claude = Claude(model, effort="low", max_tokens=8192)
    batches = [todo[i : i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]

    import anthropic

    def run(batch: list[str]) -> list[dict]:
        user = json.dumps([{"id": i, "headline": t} for i, t in enumerate(batch)])
        try:
            out = claude.json_call(system, user, schema, name="headline-teacher", metadata={"batch_size": len(batch)}).parsed
        except (anthropic.APIError, RuntimeError, ValueError, KeyError) as exc:  # one bad batch must not lose the others
            print(f"  batch failed: {exc}")
            return []
        got = {r["id"]: r["label"] for r in out["labels"] if 0 <= r["id"] < len(batch)}
        return [{"key": text_key(batch[i]), "label": lab} for i, lab in got.items()]

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool, CACHE_FILE.open("a") as fh:
        for n, rows in enumerate(pool.map(run, batches), 1):
            for row in rows:
                fh.write(json.dumps(row) + "\n")
                cache[row["key"]] = row["label"]
            if n % 20 == 0:
                print(f"  {n}/{len(batches)} batches, ${claude.usage.cost_usd:.3f}", flush=True)
    claude.flush()
    u = claude.usage
    print(f"teacher usage: {u.calls} calls, {u.input_tokens:,} in / {u.output_tokens:,} out tokens, ${u.cost_usd:.4f}")
    return cache


def review_sample(df: pd.DataFrame, seed: int = 7) -> set[str]:
    """REVIEW_SIZE ids spread as evenly as possible over labels."""
    rng = random.Random(seed)
    groups = {lab: list(g["headline_id"]) for lab, g in df.groupby("label")}
    for ids in groups.values():
        rng.shuffle(ids)
    picked: list[str] = []
    while len(picked) < min(REVIEW_SIZE, len(df)):
        for ids in groups.values():
            if ids and len(picked) < REVIEW_SIZE:
                picked.append(ids.pop())
    return set(picked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="no API calls; label with keyword rules")
    parser.add_argument("--model", default=config.TEACHER_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="label at most N headlines (random, cases first)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    df = build_headlines()
    case_texts: set[str] = set()
    cases_file = config.OUT / "cases.jsonl"
    if cases_file.exists():
        case_texts = {json.loads(line)["headline"] for line in cases_file.open()} - {None}
    df["in_case"] = df["text"].isin(case_texts)
    if args.limit and args.limit < len(df):
        rest = df[~df["in_case"]].sample(max(0, args.limit - int(df["in_case"].sum())), random_state=1)
        df = pd.concat([df[df["in_case"]], rest]).sort_values("ts").reset_index(drop=True)

    if args.dry_run:
        df["label"] = df["text"].map(rule_label)
        df["label_source"] = "rules"
    else:
        cache = teacher_label(list(df["text"]), args.model, args.workers)
        df["label"] = df["text"].map(lambda t: cache.get(text_key(t)))
        missing = int(df["label"].isna().sum())
        if missing:
            print(f"warning: {missing} headlines without a teacher label are dropped (re-run to retry)")
            df = df.dropna(subset=["label"])
        df["label_source"] = f"teacher:{args.model}"

    shock = set(load_task()["fundamental_shock_labels"])
    df["fundamental_shock"] = df["label"].isin(shock)
    review = review_sample(df)
    df["review"] = df["headline_id"].isin(review)

    config.OUT.mkdir(parents=True, exist_ok=True)
    df[["headline_id", "split", "ts", "ticker", "pair", "role", "title", "text", "label", "publisher", "url"]].to_csv(
        config.INTERIM / "headlines_private.csv", index=False
    )
    with (config.OUT / "headlines.jsonl").open("w") as fh:
        for r in df.to_dict("records"):
            fh.write(
                json.dumps(
                    {
                        "headline_id": r["headline_id"],
                        "split": r["split"],
                        "text": r["text"],
                        "company_role": r["role"],
                        "label": r["label"],
                        "fundamental_shock": bool(r["fundamental_shock"]),
                        "label_source": r["label_source"],
                        "review": bool(r["review"]),
                        "in_case": bool(r["in_case"]),
                    }
                )
                + "\n"
            )
    sample = df[df["review"]][["headline_id", "split", "text", "label", "fundamental_shock"]].copy()
    sample["correct_label"] = ""
    sample["note"] = ""
    sample.to_csv(config.OUT / "headline_review.csv", index=False)

    stats = df.groupby("split")["label"].value_counts().unstack(fill_value=0).T
    stats["total"] = stats.sum(axis=1)
    print(stats.to_string())
    print(f"\n{len(df)} headlines ({df['label_source'].iloc[0]}), fundamental_shock share {df['fundamental_shock'].mean():.1%}")


if __name__ == "__main__":
    main()
