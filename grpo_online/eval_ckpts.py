"""Evaluate online-GRPO checkpoints (and the base model) on a fixed AppWorld split.

One vLLM server (base + LoRA hot-swap, see vllm_gen.py) serves every checkpoint in
turn; each checkpoint runs `appworld run` greedily (temperature 0) on the split and
we aggregate AppWorld's per-task evaluation into TGC / SGC:
  TGC = fraction of tasks fully solved (all unit tests pass)
  SGC = fraction of scenarios (task ids sharing the prefix before "_") whose tasks
        are all solved
Usage (from the repo root, trainer venv):
  .venv/bin/python -m grpo_online.eval_ckpts --config grpo_online/config/online_v1_vllm.yaml \
      --gpu 0 --dataset dev --ckpts base grpo_online/runs/v1_vllm/ckpt_round_30 ...
  --ckpts all  -> base + every ckpt_round_* under the run's output_dir (ascending)
Results: <output_dir>/eval/<dataset>/<name>/evaluations/<dataset>.json (AppWorld format)
and a summary table <output_dir>/eval/<dataset>/summary.json + .md.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import requests

from grpo_online.config import load_config
from grpo_online.vllm_gen import VllmGenServer

logger = logging.getLogger("online")
EXPERIMENT = "eval/qwen35_9b_greedy"  # appworld config name (appworld_configs/eval/...)


def tgc_sgc(individual: dict) -> dict:
    """AppWorld task/scenario goal completion from an evaluations/<ds>.json 'individual' dict."""
    solved = {t: bool(r.get("success")) for t, r in individual.items()}
    scen: dict[str, list[bool]] = {}
    for t, ok in solved.items():
        scen.setdefault(t.rsplit("_", 1)[0], []).append(ok)
    n = len(solved)
    return {
        "n_tasks": n,
        "tgc": (sum(solved.values()) / n) if n else 0.0,
        "sgc": (sum(all(v) for v in scen.values()) / len(scen)) if scen else 0.0,
        "n_scenarios": len(scen),
        "mean_test_pass_frac": (sum(
            (len(r.get("passes", [])) / int(r["num_tests"])) if int(r.get("num_tests", 0)) else 0.0
            for r in individual.values()) / n) if n else 0.0,
    }


def ckpt_sort_key(p: str) -> int:
    m = re.search(r"ckpt_round_(\d+)", p)
    return int(m.group(1)) if m else -1


def run_one(name: str, model_name: str, dataset: str, appworld_bin: str, appworld_root: Path,
            port: int, nproc: int, out_root: Path) -> dict:
    exp = f"{EXPERIMENT}_{dataset}_{name}"
    # Per-run experiment stub importing the base eval config (same trick as rollout.py).
    cfg_dir = appworld_root / "experiments" / "configs"
    stub = cfg_dir / f"{exp}.jsonnet"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text('import "qwen35_9b_greedy.jsonnet"\n')
    out_dir = appworld_root / "experiments" / "outputs" / exp
    if out_dir.exists():
        shutil.rmtree(out_dir)
    env = {**os.environ,
           "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "dummy"), "NO_API_KEY": "dummy",
           "EVAL_MODEL_NAME": model_name, "EVAL_DATASET": dataset,
           "VLLM_PORT": str(port), "MODEL_SERVER_URL": f"http://localhost:{port}"}
    t0 = time.time()
    subprocess.run([appworld_bin, "run", exp, "--num-processes", str(nproc),
                    "--with-evaluation", "--without-setup"],
                   env=env, check=False, cwd=str(appworld_root))
    dst = out_root / name
    if dst.exists():
        shutil.rmtree(dst)
    if not out_dir.exists():
        raise RuntimeError(f"appworld run produced no output for {exp}")
    out_dir.rename(dst)
    ev = sorted((dst / "evaluations").glob("*.json"))
    if not ev:
        raise RuntimeError(f"no evaluations/*.json under {dst}")
    individual = json.loads(ev[0].read_text()).get("individual", {})
    res = {"name": name, "model": model_name, **tgc_sgc(individual), "seconds": round(time.time() - t0)}
    logger.info("eval %s (%s): TGC %.3f SGC %.3f test-pass %.3f n=%d in %ds", name, dataset,
                res["tgc"], res["sgc"], res["mean_test_pass_frac"], res["n_tasks"], res["seconds"])
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", required=True, help="CUDA device for the vLLM server")
    ap.add_argument("--port", type=int, default=8102)
    ap.add_argument("--dataset", default="dev")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--appworld-bin", default=".appworld.venv/bin/appworld")
    ap.add_argument("--ckpts", nargs="+", default=["all"],
                    help="'base', checkpoint dirs, or 'all' (= base + every ckpt_round_* in output_dir)")
    ap.add_argument("--attach-port", type=int, default=None,
                    help="evaluate against an ALREADY RUNNING vLLM server on this port "
                         "instead of starting one (in-run validation). The checkpoint is "
                         "loaded under --lora-name, i.e. the second LoRA slot, so the "
                         "training loop's own adapter is untouched.")
    ap.add_argument("--lora-name", default="eval_probe",
                    help="LoRA slot name to use with --attach-port")
    ap.add_argument("--out-name", default=None,
                    help="results subdir under <output_dir>/eval/ (default: the dataset name); "
                         "use distinct names when sharding checkpoints across GPUs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_root = Path(cfg.output_dir) / "eval" / (args.out_name or args.dataset)
    out_root.mkdir(parents=True, exist_ok=True)
    appworld_root = Path.cwd() / "appworld"
    appworld_bin = str(Path(args.appworld_bin).resolve())

    targets: list[tuple[str, str | None]] = []  # (name, adapter dir or None for base)
    for c in args.ckpts:
        if c == "all":
            targets.append(("base", None))
            for d in sorted(glob_ckpts(cfg.output_dir), key=ckpt_sort_key):
                targets.append((Path(d).name, d))
        elif c == "base":
            targets.append(("base", None))
        else:
            targets.append((Path(c.rstrip("/")).name, c))

    summary_path = out_root / "summary.json"
    done = {r["name"]: r for r in json.loads(summary_path.read_text())} if summary_path.exists() else {}
    attached = args.attach_port is not None
    port = args.attach_port if attached else args.port
    srv = VllmGenServer(cfg, port=port, gpus=args.gpu, log_path=str(out_root / "vllm.log"),
                        lora_name=args.lora_name if attached else None)
    try:
        if attached:
            srv.wait_ready(timeout=600)  # server owned by the training loop
            logger.info("eval: attached to vLLM on port %d, LoRA slot %r", port, args.lora_name)
        else:
            srv.start()
        for name, adapter in targets:
            if name in done:
                logger.info("eval %s: already done, skipping", name)
                continue
            if adapter is None:
                model_name = cfg.vllm_base_served_name
            else:
                srv.load_adapter(adapter)
                model_name = srv.lora_name
            done[name] = run_one(name, model_name, args.dataset, appworld_bin, appworld_root,
                                 port, args.num_processes, out_root)
            rows = sorted(done.values(), key=lambda r: (r["name"] != "base", ckpt_sort_key(r["name"])))
            summary_path.write_text(json.dumps(rows, indent=1))
            (out_root / "summary.md").write_text(
                f"| checkpoint | TGC | SGC | mean test-pass | n |\n|---|---|---|---|---|\n" +
                "\n".join(f"| {r['name']} | {r['tgc']:.3f} | {r['sgc']:.3f} | {r['mean_test_pass_frac']:.3f} | {r['n_tasks']} |"
                          for r in rows) + "\n")
    finally:
        if attached:
            # Free the second LoRA slot; leave the loop's server running.
            try:
                requests.post(f"http://localhost:{port}/v1/unload_lora_adapter",
                              json={"lora_name": args.lora_name}, timeout=120)
            except Exception as e:
                logger.warning("eval: could not unload %s: %s", args.lora_name, e)
        else:
            srv.stop()
    print((out_root / "summary.md").read_text())


def glob_ckpts(output_dir: str) -> list[str]:
    """Checkpoint dirs of a run, excluding the `_vllm` key-remapped exports that
    load_adapter writes next to each one (see vllm_gen.export_adapter_for_vllm)."""
    return [str(p) for p in Path(output_dir).glob("ckpt_round_*")
            if p.is_dir() and not p.name.endswith("_vllm")]


if __name__ == "__main__":
    main()
