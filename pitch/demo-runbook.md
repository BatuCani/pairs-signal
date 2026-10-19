# News Gate - demo runbook (1 minute, 5 steps)

1. Fill the `numbers` block in `pitch/slides.html` from the `uv run experiment --task news_event ... --split test` summaries (teacher-test, slm-v1-test, slm-v2-test) and `pairs_signal/README.md`; nothing may stay pink TBD.
2. Open `pitch/slides.html`, press **F**; second screen or **N** shows the speaker notes (= `script.md`).
3. Optional live proof if time allows, from `pairs_signal/` with keys sourced: `uv run python ../pitch/news_gate.py '$KO: Coca-Cola announces that John Murphy will become president and chief financial officer effective Oct. 1'` (single quotes), then the `news-gate-live` trace in Langfuse.
4. If Runpod hangs: `export STUDENT_BASE_URL=https://inference.distillabs.ai/v1 STUDENT_MODEL=<endpoint> STUDENT_API_KEY=<key>` (distil-hosted endpoint, setup in `pairs_signal/README.md`), or skip the live part; backup = screen recording of step 3 made during warm-up.
5. After the pitch: `runpodctl serverless delete <endpoint-id>` and delete the distil-hosted endpoint, so no GPU keeps billing.
