# News Gate demo UI

Live demo for the News Gate classifier: a headline goes in, the distil labs model (Qwen3.5-0.8B) returns one of 10 event types, and the UI shows whether it is a real shock or just noise.

## Start

```bash
uv run ui/server.py
```

Then open http://localhost:8000.

Without any model endpoint configured the server runs in **Mock** mode (keyword rules, marked "Mock" in the UI), so the page always works.

## Configuration (environment only)

| Variable | Meaning |
| --- | --- |
| `MODEL_BASE_URL`, `MODEL_API_KEY`, `MODEL_NAME` | One OpenAI-compatible endpoint (distil-hosted or Runpod vLLM). |
| `MODEL_V1_BASE_URL`, `MODEL_V1_API_KEY`, `MODEL_V1_NAME` | Version 1 (1,500 real examples). |
| `MODEL_V2_BASE_URL`, `MODEL_V2_API_KEY`, `MODEL_V2_NAME` | Version 2 (plus 1,000 synthetic). When V1 and/or V2 are set they replace `MODEL_*` and are shown side by side. |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | Langfuse tracing via `langfuse.openai`; each call returns an "In Langfuse ansehen" link. |
| `PORT` | Server port, default 8000. |

The server also reads `~/.config/pairs-signal.env` (never commit keys); variables already set in the shell win.

Example with both versions:

```bash
MODEL_V1_BASE_URL=https://.../v1 MODEL_V1_API_KEY=... MODEL_V1_NAME=news-gate-v1 \
MODEL_V2_BASE_URL=https://.../v1 MODEL_V2_API_KEY=... MODEL_V2_NAME=news-gate-v2 \
uv run ui/server.py
```

## Prompt and data

The system prompt is the `task_description` from `job_description.json`, exactly as used for training, and the headline is the user message.
The server reads `pairs_signal/data/out/distil/news_event/` when it exists in the checkout and otherwise the snapshot in `ui/data/` (same files).
The live feed classifies one test headline per label from `test.jsonl` and scores it against the human label.

## API

- `POST /classify` with `{"headline": "..."}` returns `{"results": [{variant, model, label, shock, latency_ms, trace_url, mock, error}]}`.
- `GET /config`, `GET /feed`.
