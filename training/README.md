# Training and evaluation material

Everything that ran on distil labs and Langfuse on the hackathon evening (2026-10-19), as far as it is reproducible without secrets.
The dataset these jobs started from is committed in `pairs_signal/data/out/distil/news_event/` (`train.jsonl` with 1,500 rows, `test.jsonl` with 400 rows, `config.yaml`, `job_description.json`) and is not duplicated here.

## Files

| File | What it is |
| --- | --- |
| `config-fast-train.yaml` | distil labs job config for **v1**: the `news_event/config.yaml` plus a tuning block (2 epochs, batch size 8, learning rate 1e-4). Student Qwen3.5-0.8B, teacher and judge GLM 5.3 (`zai.glm-5.3-low-thinking`). |
| `config-flash.yaml` | distil labs job config for the **v2** synthetic data: the faster `zai.glm-5.3-flash-low-thinking` teacher generates about 1,000 new training headlines, one adaptive mutator spreads them over the 10 labels. It sets no tuning block. |
| `v2-synthetic/train.jsonl` | The v2 training set as downloaded from distil labs: the 1,500 seed rows plus 1,022 teacher-generated synthetic headlines (2,522 rows, distil `messages` format). |
| `preds-v1/preds-v1.jsonl` | The distil labs evaluation of v1 on the 400 held-out test headlines: prompt, dataset label (`completion`), model answer (`prediction`) and per-row scores. Exact match over this file is 88.0 %, shock vs. noise 94.75 %. |
| `lf_experiment.py` | Langfuse experiment: runs the deployed v1 endpoint over the Langfuse dataset `news-event-test` with two evaluators, `exact_match` and `shock_vs_noise`. This is the source of the 88.5 % / 95.0 % live numbers. |
| `watch.sh` | Polls a distil labs job every 30 s and appends its status to `progress.log`. |
| `autodeploy.sh` | Waits for a training job, then creates a deployment and an inference endpoint and links an API key to it. |
| `finish-endpoint.sh` | Finishes a deployment that was already created: waits for `JOB_SUCCESS`, opens the endpoint and links the key. |

The summary of `preds-v1.jsonl` (per-label accuracy, top confusions, example errors) is `ui/data/insights-v1.json`, which the Results page of the demo reads.
"Shock" means one of `earnings`, `m_and_a`, `legal_regulatory`, `management_change`; every other label is noise.

## Order the steps ran in

All times are CEST on 2026-10-19, taken from the job records.
Commands use the distil labs CLI 0.35 (`distil`); see `pairs_signal/README.md` for login and the full command reference.

1. 20:03 - Upload the dataset: `distil dataset create --data pairs_signal/data/out/distil/news_event`.
2. 20:03 - Teacher baseline: `distil teacher-evaluation create-from-dataset <dataset-id>` (GLM 5.3: 79.25 % on the 400 test headlines).
3. 20:05 - v2 data: synthetic generation from the same dataset with `config-flash.yaml` (`distil dataset generate-synthetic-data-train`), result downloaded as `v2-synthetic/train.jsonl`.
4. 20:10 - v2 student: `distil slm create-from-dataset` on the synthetic dataset (90.5 % exact match).
5. 20:17 - v1 student: `distil slm create-from-dataset` on the uploaded 1,500-row dataset with `config-fast-train.yaml`; `watch.sh slm <slm-id> train-fast` tracked it to `JOB_SUCCESS` at 20:53 (about 35 minutes for the whole job, 6.6 minutes of it GPU training).
6. 20:37 to 21:04 - Serve v1: `autodeploy.sh <slm-id> news-gate-batu-v1` started the deployment, `finish-endpoint.sh <deployment-id>` opened the OpenAI-compatible endpoint on `https://inference.distillabs.ai/v1`.
7. 20:55 - Download the v1 test predictions to `preds-v1/` and summarize them into `insights-v1.json`.
8. 21:07 - Live check in Langfuse: `uv run training/lf_experiment.py`.

The Langfuse dataset `news-event-test` holds the same 400 rows as `test.jsonl` (`input.headline`, label as expected output); it was uploaded by hand on the evening, so that step has no script here.

## Credentials

Nothing in this folder contains a key.
The shell scripts take the CLI from `DISTIL_BIN` (default `distil` on `PATH`), write logs to `WORK_DIR` (default the current directory) and link the key named in `DISTIL_API_KEY_NAME`.
`lf_experiment.py` reads `DISTIL_API_KEY`, `NEWS_GATE_MODEL` and the standard `LANGFUSE_*` variables.
distil labs inference endpoints are temporary, so rerunning steps 6 to 8 later needs a freshly deployed model.
