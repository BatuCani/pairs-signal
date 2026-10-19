"""Stream FNSPID news from Hugging Face and keep only headlines of the pair tickers.

The default source CSV is ~5.7 GB; it is parsed while streaming so only the small filtered file
lands in data/raw. A dropped connection resumes with an HTTP Range request from the last
complete CSV record, and the run fails unless every byte of the source was read.
Re-running skips the download when the output exists.
"""

import argparse
import csv
import sys
import time

import requests

from pairs_signal import config

KEEP = ["date", "ticker", "title", "publisher", "url"]
COLUMNS = ("Date", "Article_title", "Stock_symbol", "Publisher", "Url")
CHUNK = 1 << 20
MAX_RETRIES = 8


class LineSource:
    """Byte lines from a ranged HTTP stream; tracks how many bytes the consumer has taken."""

    def __init__(self, url: str, start: int, end: int | None):
        self.url, self.offset, self.end = url, start, end

    def lines(self):
        rng = f"bytes={self.offset}-" + (str(self.end - 1) if self.end else "")
        with requests.get(self.url, stream=True, headers={"Range": rng}, timeout=60) as resp:
            resp.raise_for_status()
            buf = b""
            for chunk in resp.iter_content(CHUNK):
                buf += chunk
                *complete, buf = buf.split(b"\n")
                for line in complete:
                    self.offset += len(line) + 1
                    yield line.decode("utf-8", errors="replace") + "\n"
            if buf and self.end:  # a ranged smoke test ends mid-line
                return
            if buf:
                self.offset += len(buf)
                yield buf.decode("utf-8", errors="replace")


def content_length(url: str) -> int:
    resp = requests.head(url, allow_redirects=True, timeout=60)
    resp.raise_for_status()
    return int(resp.headers["Content-Length"])


def stream_filter(url: str, tickers: set[str], out_path, max_bytes: int | None) -> int:
    csv.field_size_limit(sys.maxsize)
    total = max_bytes or content_length(url)
    tmp = out_path.with_suffix(".partial")
    kept, scanned, record_end, header, idx = 0, 0, 0, None, {}
    started = time.time()
    with tmp.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(KEEP)
        for attempt in range(MAX_RETRIES + 1):
            source = LineSource(url, record_end, max_bytes)
            reader = csv.reader(source.lines())
            try:
                for row in reader:
                    # csv.reader pulls lines lazily, so after a row the source offset is
                    # exactly the end of that record: a safe point to resume from.
                    record_end = source.offset
                    if header is None:
                        header = row
                        idx = {name: header.index(name) for name in COLUMNS}
                        continue
                    scanned += 1
                    if len(row) == len(header):
                        ticker = row[idx["Stock_symbol"]].strip().upper()
                        if ticker in tickers:
                            writer.writerow([row[idx["Date"]], ticker, row[idx["Article_title"]].strip(), row[idx["Publisher"]], row[idx["Url"]]])
                            kept += 1
                    if scanned % 1_000_000 == 0:
                        mins = (time.time() - started) / 60
                        print(f"  {scanned:,} rows, {record_end / 1e9:.2f}/{total / 1e9:.2f} GB, {kept:,} kept, {mins:.1f} min", flush=True)
            except (requests.RequestException, csv.Error) as exc:
                print(f"  stream dropped at {record_end:,} bytes ({exc.__class__.__name__}), resuming", flush=True)
                time.sleep(min(2**attempt, 30))
                continue
            break
    if max_bytes is None and record_end != total:
        raise SystemExit(f"read {record_end:,} of {total:,} bytes; partial output left in {tmp}")
    tmp.replace(out_path)
    print(f"  {scanned:,} rows, {record_end:,} bytes read")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-bytes", type=int, default=None, help="only read the first N bytes (smoke test)")
    parser.add_argument("--force", action="store_true", help="re-download even if the output exists")
    parser.add_argument("--source", choices=sorted(config.FNSPID_SOURCES), default="external", help="external: 5.7 GB, to 2020-06; nasdaq: 23 GB, to 2023")
    args = parser.parse_args()
    url = config.FNSPID_SOURCES[args.source]

    out = config.NEWS_FILE
    if out.exists() and not args.force:
        print(f"{out} exists, skipping (use --force to refresh)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Streaming {url} for {len(config.TICKERS)} tickers")
    kept = stream_filter(url, set(config.TICKERS), out, args.max_bytes)
    print(f"Wrote {kept:,} headlines to {out}")


if __name__ == "__main__":
    main()
