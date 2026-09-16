#!/usr/bin/env bash
# SeRPO Round-1 v3-G on Qwen3.5-9B — cache-fix (v3-F) + per-token KL cap.
# Reuses the v3-F micro=1 pre-cache (no recache). Expect step 0 kl=0 and a
# bounded step 1 (no more seq-10 outlier-token blowup).
set -euo pipefail
cd "$(dirname "$0")/../.."
source /data/minjeong/.envs/grpo9b/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python -m grpo.trainer.run_grpo --config grpo/trainer/config/serpo_avg_9b_v3i.yaml
