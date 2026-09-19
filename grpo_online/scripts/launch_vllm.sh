#!/usr/bin/env bash
# Online GRPO with the vLLM generation backend. 2 GPUs: trainer on $TRAINER_GPU,
# vLLM (started by run_online itself) on the config's gen_gpus.
#
# Usage: bash grpo_online/scripts/launch_vllm.sh [config.yaml] [--resume-round N]
# Env:   TRAINER_GPU (default 0), APPWORLD_BIN (default .appworld.venv/bin/appworld)
set -euo pipefail
cd "$(dirname "$0")/../.."
CFG="${1:-grpo_online/config/online_v1_vllm.yaml}"; shift || true
TRAINER_GPU="${TRAINER_GPU:-0}"
APPWORLD_BIN="${APPWORLD_BIN:-$PWD/.appworld.venv/bin/appworld}"
[ -f .env ] && set -a && source .env && set +a
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set (rubric judge)}"
OUT=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CFG")
mkdir -p "$OUT"
# Install the tracked AppWorld experiment configs into the (gitignored) appworld tree.
mkdir -p appworld/experiments/configs && cp -r grpo_online/appworld_configs/. appworld/experiments/configs/
python3 grpo_online/appworld_patches.py
# Trainer + loop (+ vLLM child on gen_gpus; CUDA_VISIBLE_DEVICES is overridden for it).
CUDA_VISIBLE_DEVICES="$TRAINER_GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -m grpo_online.run_online --config "$CFG" --appworld-bin "$APPWORLD_BIN" "$@" \
  2>&1 | tee -a "$OUT/loop.log"
