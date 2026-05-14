"""Evaluate a trained LoRA on AppWorld test_normal.

Steps:
  1. Register the LoRA adapter with the running vLLM (POST /v1/load_lora_adapter).
  2. Invoke ``appworld run`` against test_normal pointing at the LoRA-named model.
  3. Parse evaluations/test_normal.json → aggregate success rate.
  4. Write a metrics file.

Prerequisites:
  - vLLM is running externally (e.g., via serve_qwen_vllm.sh) with
    --enable-lora and a free slot in --max-loras.
  - The AppWorld config used by the experiment name must reference the
    LoRA-named model (so the env's chat completion calls route to it).

The LoRA adapter dir must be the directory peft saved via ``model.save_pretrained``,
containing ``adapter_config.json`` and ``adapter_model.safetensors``.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def register_lora(vllm_url: str, lora_name: str, lora_path: Path) -> None:
    """POST to vLLM's load_lora_adapter endpoint."""
    payload = {"lora_name": lora_name, "lora_path": str(lora_path)}
    r = requests.post(
        f"{vllm_url}/v1/load_lora_adapter", json=payload, timeout=120,
    )
    r.raise_for_status()
    logger.info("registered LoRA %s ← %s", lora_name, lora_path)


def unregister_lora(vllm_url: str, lora_name: str) -> None:
    """Best-effort unload — logs but does not raise on failure."""
    try:
        r = requests.post(
            f"{vllm_url}/v1/unload_lora_adapter",
            json={"lora_name": lora_name},
            timeout=60,
        )
        r.raise_for_status()
    except Exception as e:
        logger.warning("unregister LoRA %s failed: %s", lora_name, e)


def run_appworld_test(
    experiment_name: str,
    agent_name: str,
    num_processes: int,
    appworld_root: Path,
    dataset_name: str = "test_normal",
) -> Path:
    """Run ``appworld run``. Returns path to evaluations/<dataset>.json."""
    cmd = [
        "appworld", "run", experiment_name,
        "--agent-name", agent_name,
        "--dataset-name", dataset_name,
        "--num-processes", str(num_processes),
        "--with-evaluation",
    ]
    logger.info("running: %s (cwd=%s)", " ".join(cmd), appworld_root)
    subprocess.run(cmd, cwd=appworld_root, check=True)
    return (
        appworld_root
        / "experiments"
        / "outputs"
        / experiment_name
        / "evaluations"
        / f"{dataset_name}.json"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="LoRA adapter dir (peft save_pretrained output)")
    ap.add_argument("--method", choices=["vanilla", "serpo"], required=True)
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--appworld-root", type=Path,
                    default=Path("/data/minjeong/Autonomous_agent/appworld"))
    ap.add_argument("--agent-name", default="simplified_react_code_agent")
    ap.add_argument("--num-processes", type=int, default=4)
    ap.add_argument("--dataset-name", default="test_normal")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    lora_name = f"qwen25_7b_lora_{args.method}_round1"
    experiment_name = f"eval/{args.dataset_name}/{lora_name}"

    register_lora(args.vllm_url, lora_name, args.ckpt.resolve())
    try:
        eval_path = run_appworld_test(
            experiment_name,
            args.agent_name,
            args.num_processes,
            args.appworld_root,
            dataset_name=args.dataset_name,
        )
        data = json.loads(eval_path.read_text())
        aggregate = data.get("aggregate", {})
        success_rate = aggregate.get("task_goal_completion")
        per_task = {
            tid: rec.get("success", False)
            for tid, rec in data.get("individual", {}).items()
        }
        out = {
            "method": args.method,
            "dataset": args.dataset_name,
            "lora_name": lora_name,
            "ckpt": str(args.ckpt),
            "success_rate": success_rate,
            "n_tasks": len(per_task),
            "n_success": sum(per_task.values()),
            "per_task": per_task,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2))
        logger.info(
            "wrote %s — success_rate=%s (%d/%d)",
            args.output, success_rate, sum(per_task.values()), len(per_task),
        )
    finally:
        unregister_lora(args.vllm_url, lora_name)


if __name__ == "__main__":
    main()
