# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.40", "langfuse>=3,<4"]
# ///
"""Langfuse experiment: run the v1 student over the news-event-test dataset and score it.

Evaluators: exact_match (label equals the dataset label) and shock_vs_noise (both sides agree on shock).
env: DISTIL_API_KEY (distil labs inference key), NEWS_GATE_MODEL (endpoint name, default news-gate-batu-v1-3sY4Ma),
     LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST.
Run: uv run training/lf_experiment.py
"""
import json, os
from pathlib import Path
from langfuse import get_client, Evaluation
from langfuse.openai import OpenAI
JD = Path(__file__).resolve().parents[1] / "pairs_signal/data/out/distil/news_event/job_description.json"
KEY = os.environ["DISTIL_API_KEY"]
MODEL = os.environ.get("NEWS_GATE_MODEL", "news-gate-batu-v1-3sY4Ma")
SYS = json.loads(JD.read_text())["task_description"]
client = OpenAI(base_url="https://inference.distillabs.ai/v1", api_key=KEY)
SHOCK = {"earnings", "m_and_a", "legal_regulatory", "management_change"}
def task(*, item, **kw):
    r = client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": SYS}, {"role": "user", "content": item.input["headline"]}], temperature=0, max_tokens=16, extra_body={"chat_template_kwargs": {"enable_thinking": False}}, name="news-gate-v1")
    return r.choices[0].message.content.strip()
def exact(*, output, expected_output, **kw):
    return Evaluation(name="exact_match", value=float(output == expected_output))
def shock(*, output, expected_output, **kw):
    return Evaluation(name="shock_vs_noise", value=float((output in SHOCK) == (expected_output in SHOCK)))
lf = get_client()
ds = lf.get_dataset("news-event-test")
res = ds.run_experiment(name="news-gate-v1 (Qwen3.5-0.8B, distil labs)", description="Trained student on 400 test headlines", task=task, evaluators=[exact, shock], max_concurrency=8)
lf.flush()
print(res.format()[-1500:])
