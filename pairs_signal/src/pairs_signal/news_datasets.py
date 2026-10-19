"""Stock-news classification datasets from public labeled sources.

  news_event      zeroshot/twitter-financial-news-topic (MIT), 20 topics mapped to 10 event
                  types (tasks/news_event_labels.json documents the mapping)
  news_sentiment  zeroshot/twitter-financial-news-sentiment (MIT), Bearish/Bullish/Neutral
                  mapped to negative/positive/neutral

Splits: the source validation split is the fixed test set; val is a seeded 10% of the source
train split. Texts are cleaned (t.co links, whitespace) and deduplicated on a normalized form;
a text that is in test is removed from train and val.

--phrasebank also downloads Financial PhraseBank (takala/financial_phrasebank, CC BY-NC-SA 3.0,
sentences where all 5-8 annotators agree) as an extra human-labeled sentiment test set. It is
written to data/out/sets/news_sentiment_phrasebank_test.jsonl, which is gitignored because of
the share-alike license.

Outputs: data/raw/hf/*.csv (downloads), data/out/news/<task>.jsonl (all mapped rows).
Run export_distil afterwards for the balanced sets and distil labs bundles.
"""

import argparse
import io
import json
import random
import re
import zipfile

import pandas as pd
import requests

from pairs_signal import config

HF = "https://huggingface.co/datasets"
SOURCES = {
    "news_event": ("zeroshot/twitter-financial-news-topic", "topic_train.csv", "topic_valid.csv"),
    "news_sentiment": ("zeroshot/twitter-financial-news-sentiment", "sent_train.csv", "sent_valid.csv"),
}
# Integer labels of the source CSVs, from the dataset cards.
SOURCE_NAMES = {
    "news_event": [
        "Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt", "Dividend",
        "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion", "Gold | Metals | Materials",
        "IPO", "Legal | Regulation", "M&A | Investments", "Macro", "Markets", "Politics", "Personnel Change",
        "Stock Commentary", "Stock Movement",
    ],
    "news_sentiment": ["Bearish", "Bullish", "Neutral"],
}
PHRASEBANK_ZIP = f"{HF}/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
HF_RAW = config.RAW / "hf"
VAL_SHARE = 0.10
SEED = 21


def download(repo: str, name: str) -> pd.DataFrame:
    path = HF_RAW / repo.replace("/", "__") / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(f"{HF}/{repo}/resolve/main/{name}", timeout=60)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return pd.read_csv(path)


def clean(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9$%]+", " ", text.lower()).strip()


def build(task: str) -> pd.DataFrame:
    repo, train_file, test_file = SOURCES[task]
    mapping = json.loads((config.TASKS / f"{task}_labels.json").read_text())["source_mapping"]["map"]
    names = SOURCE_NAMES[task]
    frames = []
    for file, split in ((train_file, "train"), (test_file, "test")):
        df = download(repo, file)
        df["source_label"] = df["label"].map(lambda i: names[int(i)])
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].astype(str).map(clean)
    df["label"] = df["source_label"].map(mapping)
    df = df[df["label"].notna() & (df["text"].str.len() >= 15)].copy()
    df["key"] = df["text"].map(norm)
    # Keep one row per normalized text; test wins so it never overlaps train/val.
    df = df.sort_values("split", key=lambda s: s.map({"test": 0, "train": 1})).drop_duplicates("key")
    rng = random.Random(SEED)
    train_idx = list(df.index[df["split"] == "train"])
    rng.shuffle(train_idx)
    df.loc[train_idx[: int(len(train_idx) * VAL_SHARE)], "split"] = "val"
    df = df.sort_index().reset_index(drop=True)
    df["id"] = [f"{task[5:7]}{i:05d}" for i in range(len(df))]
    df["source"] = repo
    return df[["id", "split", "text", "label", "source_label", "source"]]


def phrasebank() -> pd.DataFrame:
    path = HF_RAW / "takala__financial_phrasebank" / "FinancialPhraseBank-v1.0.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(PHRASEBANK_ZIP, timeout=60)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    with zipfile.ZipFile(path) as zf:
        member = next(n for n in zf.namelist() if n.endswith("Sentences_AllAgree.txt"))
        lines = io.TextIOWrapper(zf.open(member), encoding="latin-1").read().splitlines()
    rows = [line.rsplit("@", 1) for line in lines if "@" in line]
    df = pd.DataFrame(rows, columns=["text", "label"])
    df["text"] = df["text"].map(clean)
    return df.drop_duplicates("text").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phrasebank", action="store_true", help="also build the PhraseBank sentiment test set (not committed)")
    args = parser.parse_args()
    out = config.OUT / "news"
    out.mkdir(parents=True, exist_ok=True)
    for task in SOURCES:
        df = build(task)
        with (out / f"{task}.jsonl").open("w") as fh:
            for r in df.to_dict("records"):
                fh.write(json.dumps(r) + "\n")
        counts = df.groupby("split")["label"].value_counts().unstack(fill_value=0).T
        print(f"[{task}] {len(df)} rows -> {out / task}.jsonl\n{counts.to_string()}\n")
    if args.phrasebank:
        pb = phrasebank()
        path = config.OUT / "sets" / "news_sentiment_phrasebank_test.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for i, r in enumerate(pb.to_dict("records")):
                fh.write(json.dumps({"id": f"pb{i:05d}", "input": r["text"], "label": r["label"], "meta": {"source": "takala/financial_phrasebank"}}) + "\n")
        print(f"[phrasebank] {len(pb)} all-agree sentences -> {path}\n{pb['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
