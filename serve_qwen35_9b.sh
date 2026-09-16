#!/usr/bin/env bash
# Launch vLLM serving Qwen3.5-9B on GPU 0 for offline segmentation use.
#
# Port 8001 chosen to avoid conflict with the training-time vLLM (port 8000).
# Reasoning parser enabled so per-request enable_thinking=true returns
# reasoning_content separately. Default chat-template behavior for Qwen3.5-9B
# is reasoning OFF, which matches A-mode.
#
# Usage:
#   bash serve_qwen35_9b.sh [--gpu GPU_ID] [--port PORT]

set -euo pipefail

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${VLLM_PORT:-8001}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"
MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-9B}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu) GPU_ID="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "============================================"
echo "vLLM serve (segmentation)"
echo "  Model         : ${MODEL}"
echo "  GPU           : ${GPU_ID}"
echo "  Port          : ${PORT}"
echo "  max-model-len : ${MAX_MODEL_LEN}"
echo "  max-num-seqs  : ${MAX_NUM_SEQS}"
echo "  gpu-mem-util  : ${GPU_MEM_UTIL}"
echo "  reasoning-parser: qwen3 (per-request controlled via enable_thinking)"
echo "============================================"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

exec vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --reasoning-parser qwen3 \
  --dtype bfloat16
