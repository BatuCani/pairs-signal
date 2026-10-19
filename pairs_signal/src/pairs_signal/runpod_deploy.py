"""Serve a distil labs student on Runpod serverless vLLM with its LoRA adapter.

Reads the downloaded model directory (`distil slm download --destination model <slm-id>`,
then `tar -xf model.tar`): config.yaml names the base model and the LoRA rank, and
model-adapter/ is the adapter. Prints, or with --execute runs:
  1. hf upload <hf-repo> model/model-adapter --private   (via uvx; needs `hf auth login`)
  2. runpodctl serverless create --hub-id <vLLM worker> ... with ENABLE_LORA / LORA_MODULES
The client then uses https://api.runpod.ai/v2/<endpoint-id>/openai/v1 with model "model"
(see student.py: export RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY).
"""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import yaml

# distil labs student name -> Hugging Face base model repo.
BASE_REPOS = {
    "Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B",
    "Qwen3.5-2B": "Qwen/Qwen3.5-2B",
    "Qwen3.5-4B": "Qwen/Qwen3.5-4B",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma-3-1b-it": "google/gemma-3-1b-it",
}


def env_flag(key: str, value: str) -> str:
    """runpodctl --env is CSV-parsed: quote values with commas or quotes (e.g. LORA_MODULES JSON)."""
    item = f"{key}={value}"
    if "," in item or '"' in item:
        item = '"' + item.replace('"', '""') + '"'
    return item


def commands(model_dir: Path, hf_repo: str, hub_id: str, gpu: str, name: str, workers_min: int) -> list[list[str]]:
    cfg = yaml.safe_load((model_dir / "config.yaml").read_text())
    student = cfg["base"]["student_model_name"]
    base = BASE_REPOS.get(student)
    if base is None:
        raise SystemExit(f"add the Hugging Face repo of {student} to BASE_REPOS")
    lora_r = int(cfg.get("tuning", {}).get("lora_r", 64))
    extra = f"--max-lora-rank {lora_r}"
    if student.startswith("Qwen3"):
        extra += " --reasoning-parser qwen3"
    env = {
        "MODEL_NAME": base,
        "ENABLE_LORA": "true",
        "LORA_MODULES": json.dumps([{"name": "model", "path": hf_repo}], separators=(",", ":")),
        "MAX_MODEL_LEN": "4096",
        "VLLM_EXTRA_ARGS": extra,
    }
    create = ["runpodctl", "serverless", "create", "--hub-id", hub_id, "--name", name, "--gpu-id", gpu, "--workers-min", str(workers_min)]
    for k, v in env.items():
        create += ["--env", env_flag(k, v)]
    return [["uvx", "--from", "huggingface_hub", "hf", "upload", hf_repo, str(model_dir / "model-adapter"), "--private"], create]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default="model", help="extracted `distil slm download` directory")
    parser.add_argument("--hf-repo", required=True, help="<hf-user>/pairs-signal-adapter")
    parser.add_argument("--hub-id", default="<vllm-hub-id>", help="from `runpodctl hub search vllm`")
    parser.add_argument("--gpu", default="NVIDIA GeForce RTX 4090", help="from `runpodctl gpu list`")
    parser.add_argument("--name", default="pairs-slm")
    parser.add_argument("--workers-min", type=int, default=0, help="1 keeps a warm worker (billed while up)")
    parser.add_argument("--execute", action="store_true", help="run the commands instead of printing them")
    args = parser.parse_args()

    for cmd in commands(Path(args.model_dir), args.hf_repo, args.hub_id, args.gpu, args.name, args.workers_min):
        print(shlex.join(cmd))
        if args.execute:
            subprocess.run(cmd, check=True)
    print("\nthen: export RUNPOD_ENDPOINT_ID=<id from the output> RUNPOD_API_KEY=...; "
          "smoke test: uv run experiment --model student --split val --client-dir model")


if __name__ == "__main__":
    main()
