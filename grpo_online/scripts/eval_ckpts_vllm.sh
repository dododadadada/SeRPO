#!/usr/bin/env bash
# Evaluate base + all checkpoints of an online run on a fixed split with one vLLM server.
# Usage: bash grpo_online/scripts/eval_ckpts_vllm.sh [config.yaml] [dataset] [gpu] [--ckpts ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
CFG="${1:-grpo_online/config/online_v1_vllm.yaml}"; DS="${2:-dev}"; GPU="${3:-0}"; shift 3 2>/dev/null || shift $# 
mkdir -p appworld/experiments/configs && cp -r grpo_online/appworld_configs/. appworld/experiments/configs/
python3 grpo_online/appworld_patches.py
OUT=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CFG")
mkdir -p "$OUT/eval/$DS"
HF_HUB_OFFLINE=1 .venv/bin/python -m grpo_online.eval_ckpts --config "$CFG" --gpu "$GPU" --dataset "$DS" "$@" \
  2>&1 | tee -a "$OUT/eval/$DS/eval.log"
