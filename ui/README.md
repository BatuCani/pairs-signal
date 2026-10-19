# News Gate demo UI

Live demo for the News Gate classifier: a headline goes in, the distil labs model (Qwen3.5-0.8B) returns one of 10 event types, and the UI shows whether it is a real shock or just noise.

## Start

```bash
uv run ui/server.py
```

Then open http://localhost:8000.

Without any model endpoint configured the server runs in **Mock** mode (keyword rules, marked "Mock mode" in the UI), so the page always works.

## Configuration (environment only)

| Variable | Meaning |
| --- | --- |
| `MODEL_BASE_URL`, `MODEL_API_KEY`, `MODEL_NAME` | One OpenAI-compatible endpoint (distil-hosted or Runpod vLLM). |
| `MODEL_V1_BASE_URL`, `MODEL_V1_API_KEY`, `MODEL_V1_NAME` | Version 1 (1,500 real examples). |
| `MODEL_V2_BASE_URL`, `MODEL_V2_API_KEY`, `MODEL_V2_NAME` | Version 2 (plus 1,000 synthetic). When V1 and/or V2 are set they replace `MODEL_*` and are shown side by side. |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | Langfuse tracing via `langfuse.openai`; each call returns an "View in Langfuse" link. |
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
The live feed classifies one test headline per label from `test.jsonl` and scores it against the dataset label.

## API

- `POST /api/classify` with `{"headline": "..."}` returns `{"results": [{variant, model, label, shock, latency_ms, trace_url, mock, error}]}`.
- `GET /api/config`, `GET /api/feed`, `GET /api/insights`.

All logic lives in `ui/api/_core.py`; `ui/server.py` (local) and the Vercel functions `ui/api/*.py` share it.

## Vercel

Deploy the `ui/` folder as the project root (`cd ui && vercel deploy`, or set Root Directory to `ui`).
`index.html` is served statically, `api/*.py` run as Python functions with `data/**` bundled (`vercel.json`), dependencies come from `requirements.txt`.
Set `MODEL_*` / `MODEL_V1_*` / `MODEL_V2_*` and `LANGFUSE_*` as Vercel environment variables; without `MODEL_*` the deployment runs in Mock mode.

## Terminal demo

```bash
uv run ui/terminal_demo.py --n 25
```

Classifies N shuffled test headlines live in the terminal (prediction vs. dataset label, shock/noise, latency, Langfuse trace link), then prints an error analysis (accuracy, mean and p95 latency, confusion pairs, weakest categories, example errors).
As the next loop step it writes the misclassified headlines as distil-format training rows to `ui/data/next_train_errors.jsonl` (git-ignored) and prints the distil commands to use them.
It shares model config, system prompt and call settings with `server.py`.
