"""News Gate core: model config, prompt, classification and the /api request handler.

Shared by the local dev server (ui/server.py) and the Vercel Python functions (ui/api/*.py).
POST /api/classify {"headline": "..."} forwards the headline to one or two
OpenAI-compatible endpoints (v1/v2) with the exact system prompt used for
training, and returns label, shock flag, latency and a Langfuse trace link.
Without any endpoint configured it answers from keyword rules ("Mock").
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = UI_DIR.parent
# Prefer the live training data in the repo; fall back to the snapshot shipped with the UI.
DATA_CANDIDATES = [REPO_DIR / "pairs_signal/data/out/distil/news_event", UI_DIR / "data"]
DATA_DIR = next(d for d in DATA_CANDIDATES if (d / "job_description.json").exists())

SYSTEM_PROMPT = json.loads((DATA_DIR / "job_description.json").read_text())["task_description"]
LABELS = [
    "earnings",
    "analyst_rating",
    "m_and_a",
    "legal_regulatory",
    "management_change",
    "dividend",
    "company_news",
    "stock_movement",
    "macro_market",
    "other",
]
# Training ran without thinking, so the chat template must not open a <think> block.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
SHOCK_LABELS = {"earnings", "m_and_a", "legal_regulatory", "management_change"}


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines without overriding variables already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip().removeprefix("export ").strip(), value.strip().strip("'\""))


load_env_file(Path.home() / ".config/pairs-signal.env")
if os.environ.get("LANGFUSE_BASE_URL") and not os.environ.get("LANGFUSE_HOST"):
    os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]

LANGFUSE_ON = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
langfuse = None
if LANGFUSE_ON:
    try:
        from langfuse import get_client
        from langfuse.openai import OpenAI

        langfuse = get_client()
    except Exception as exc:  # tracing must never break the demo
        print(f"Langfuse disabled: {exc}", file=sys.stderr)
        LANGFUSE_ON = False
if not LANGFUSE_ON:
    from openai import OpenAI


def variant_config(prefix: str) -> dict | None:
    base_url = os.environ.get(f"{prefix}BASE_URL")
    name = os.environ.get(f"{prefix}NAME")
    if not base_url or not name:
        return None
    return {"base_url": base_url, "api_key": os.environ.get(f"{prefix}API_KEY") or "none", "model": name}


VARIANTS: dict[str, dict] = {}
for key, prefix in (("v1", "MODEL_V1_"), ("v2", "MODEL_V2_")):
    cfg = variant_config(prefix)
    if cfg:
        VARIANTS[key] = cfg
if not VARIANTS and (cfg := variant_config("MODEL_")):
    VARIANTS["model"] = cfg
MOCK = not VARIANTS
CLIENTS = {k: OpenAI(base_url=v["base_url"], api_key=v["api_key"], timeout=30) for k, v in VARIANTS.items()}

MOCK_RULES = [
    ("management_change", r"\b(ceo|cfo|chief|chairman|board|appoint|names?|steps? down|resign|retire|succe)"),
    ("m_and_a", r"\b(acquir|acquisition|merger|merge|takeover|buyout|stake|deal to buy|to buy|spin-?off|divest)"),
    ("legal_regulatory", r"\b(lawsuit|sue|sued|settle|fine|probe|investigat|sec\b|ftc|doj|antitrust|regulat|court|approv)"),
    ("earnings", r"\b(earnings|eps|revenue|quarter|q[1-4]\b|results|profit|guidance|outlook|beats?|miss)"),
    ("analyst_rating", r"\b(upgrade|downgrade|price target|initiat|overweight|underweight|analyst|outperform|buy rating)"),
    ("dividend", r"\b(dividend|buyback|repurchase|payout)"),
    ("macro_market", r"\b(fed|inflation|rates?|treasury|yields?|oil|dollar|gdp|jobs report|cpi|recession|economy|tariff)"),
    ("stock_movement", r"\b(shares|stock|jumps?|falls?|soars?|plunges?|rall|slump|surges?|drops?|tumbles?)"),
    ("company_news", r"\b(launch|contract|product|layoffs?|ipo|partner|plant|unveil)"),
]


def mock_label(headline: str) -> str:
    text = headline.lower()
    for label, pattern in MOCK_RULES:
        if re.search(pattern, text):
            return label
    return "other"


def parse_label(raw: str) -> str:
    text = re.sub(r"<think>.*?(</think>|$)", "", raw or "", flags=re.S).strip().lower()
    for label in sorted(LABELS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label
    return "other"


def classify_variant(key: str, headline: str) -> dict:
    cfg = VARIANTS[key]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": headline}]
    trace_url = None
    start = time.perf_counter()
    try:
        if langfuse is not None:
            with langfuse.start_as_current_span(name=f"news-gate-{key}", input={"headline": headline}) as span:
                resp = CLIENTS[key].chat.completions.create(
                    model=cfg["model"], messages=messages, temperature=0, max_tokens=64,
                    extra_body=NO_THINKING,
                    name=f"news-gate-{key}",
                )
                raw = resp.choices[0].message.content or ""
                span.update(output=parse_label(raw), metadata={"variant": key, "model": cfg["model"]})
                try:
                    trace_url = langfuse.get_trace_url(trace_id=span.trace_id)
                except Exception:
                    trace_url = None
        else:
            resp = CLIENTS[key].chat.completions.create(
                model=cfg["model"], messages=messages, temperature=0, max_tokens=64,
                extra_body=NO_THINKING,
            )
            raw = resp.choices[0].message.content or ""
        error = None
    except Exception as exc:
        raw, error = "", f"{type(exc).__name__}: {exc}"[:300]
    latency_ms = round((time.perf_counter() - start) * 1000)
    label = parse_label(raw) if not error else None
    return {
        "variant": key,
        "model": cfg["model"],
        "label": label,
        "raw": raw[:200],
        "shock": label in SHOCK_LABELS if label else None,
        "latency_ms": latency_ms,
        "trace_url": trace_url,
        "mock": False,
        "error": error,
    }


def classify(headline: str) -> list[dict]:
    if MOCK:
        start = time.perf_counter()
        label = mock_label(headline)
        return [{
            "variant": "mock",
            "model": "Mock (keyword rules)",
            "label": label,
            "raw": label,
            "shock": label in SHOCK_LABELS,
            "latency_ms": max(1, round((time.perf_counter() - start) * 1000)),
            "trace_url": None,
            "mock": True,
            "error": None,
        }]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(VARIANTS)) as pool:
        results = list(pool.map(lambda k: classify_variant(k, headline), VARIANTS))
    if langfuse is not None:
        try:
            langfuse.flush()
        except Exception:
            pass
    return results


def feed_examples(n: int = 10) -> list[dict]:
    """One test headline per label, in a fixed order, so the feed covers every class."""
    by_label: dict[str, str] = {}
    for line in (DATA_DIR / "test.jsonl").read_text().splitlines():
        msgs = json.loads(line)["messages"]
        label = msgs[-1]["content"].strip()
        by_label.setdefault(label, msgs[0]["content"])
    order = ["earnings", "analyst_rating", "macro_market", "m_and_a", "stock_movement",
             "legal_regulatory", "dividend", "management_change", "company_news", "other"]
    return [{"headline": by_label[label], "expected": label} for label in order if label in by_label][:n]


def config_payload() -> dict:
    return {
        "mock": MOCK,
        "langfuse": LANGFUSE_ON and not MOCK,
        "variants": [{"key": k, "model": v["model"]} for k, v in VARIANTS.items()] or [{"key": "mock", "model": "Mock (keyword rules)"}],
        "labels": LABELS,
        "shock_labels": sorted(SHOCK_LABELS),
        "data_dir": str(DATA_DIR.relative_to(UI_DIR) if DATA_DIR.is_relative_to(UI_DIR) else DATA_DIR.relative_to(REPO_DIR)),
    }


class ApiHandler(BaseHTTPRequestHandler):
    """Serves /api/*; used as-is by Vercel and extended by the local server."""

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/api/config":
            self._json(config_payload())
        elif path == "/api/feed":
            self._json(feed_examples())
        elif path == "/api/insights":
            self._json(json.loads((UI_DIR / "data/insights-v1.json").read_text()))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0].rstrip("/") != "/api/classify":
            self._json({"error": "not found"}, 404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            headline = str(body.get("headline", "")).strip()
        except (ValueError, json.JSONDecodeError):
            headline = ""
        if not headline:
            self._json({"error": "headline missing"}, 400)
            return
        self._json({"headline": headline, "results": classify(headline)})

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))
