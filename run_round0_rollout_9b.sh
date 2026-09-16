#!/usr/bin/env bash
# Round-0 rollout for Qwen3.5-9B (non-thinking, HF reasoning profile).
# Prereq: vLLM serving Qwen/Qwen3.5-9B on VLLM_PORT (default 8003).
#
# Usage:
#   bash run_round0_rollout_9b.sh [--group-size G] [--num-processes N] [--temperature T]
#
# Resume-safe: seeds with an existing complete output directory are skipped.

set -euo pipefail

GROUP_SIZE=8
NUM_PROCESSES=4
TEMPERATURE=1.0
TOP_P=1.0
DATASET="train"
VLLM_PORT="${VLLM_PORT:-8003}"
EXPERIMENT_NAME="rollout/round0/qwen35_9b"

while [[ $# -gt 0 ]]; do
  case $1 in
    --group-size) GROUP_SIZE="$2"; shift 2 ;;
    --num-processes) NUM_PROCESSES="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-p) TOP_P="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --vllm-port) VLLM_PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

cd "$(dirname "$0")/appworld"

if ! curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null; then
  echo "ERROR: vLLM at http://localhost:${VLLM_PORT}/health is not reachable."
  exit 1
fi

DATASET_FILE="data/datasets/${DATASET}.txt"
if [[ ! -f "${DATASET_FILE}" ]]; then
  echo "ERROR: dataset file not found: ${DATASET_FILE}"
  exit 1
fi
EXPECTED_TASKS=$(wc -l < "${DATASET_FILE}")

OUT_BASE="experiments/outputs/rollout/round0_9b"
mkdir -p "${OUT_BASE}"

echo "============================================"
echo "Round-0 rollout (Qwen3.5-9B)"
echo "  Dataset       : ${DATASET}  (${EXPECTED_TASKS} tasks)"
echo "  Group size    : ${GROUP_SIZE}"
echo "  Num processes : ${NUM_PROCESSES}"
echo "  Temperature   : ${TEMPERATURE}"
echo "  Top-p         : ${TOP_P}"
echo "  vLLM port     : ${VLLM_PORT}"
echo "============================================"

for SEED in $(seq 1 "${GROUP_SIZE}"); do
  SEED_OUT="${OUT_BASE}/seed_${SEED}"

  if [[ -d "${SEED_OUT}/tasks" ]]; then
    DONE_TASKS=$(find "${SEED_OUT}/tasks" -mindepth 2 -maxdepth 2 -type d -name "logs" 2>/dev/null | wc -l || echo 0)
    if [[ "${DONE_TASKS}" -ge "${EXPECTED_TASKS}" ]]; then
      echo ">>> seed ${SEED}: already has ${DONE_TASKS}/${EXPECTED_TASKS} tasks, skipping"
      continue
    fi
  fi

  echo ""
  echo ">>> seed ${SEED} / ${GROUP_SIZE}"

  rm -rf "experiments/outputs/${EXPERIMENT_NAME}"

  OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}" \
  NO_API_KEY="${NO_API_KEY:-dummy}" \
  ROLLOUT_SEED="${SEED}" \
  ROLLOUT_TEMPERATURE="${TEMPERATURE}" \
  ROLLOUT_TOP_P="${TOP_P}" \
  ROLLOUT_DATASET="${DATASET}" \
  VLLM_PORT="${VLLM_PORT}" \
  appworld run "${EXPERIMENT_NAME}" \
    --num-processes "${NUM_PROCESSES}" \
    --with-evaluation \
    --without-setup

  rm -rf "${SEED_OUT}"
  mv "experiments/outputs/${EXPERIMENT_NAME}" "${SEED_OUT}"
  echo ">>> seed ${SEED} done -> ${SEED_OUT}"
done

echo ""
echo "============================================"
echo "Round-0 rollout complete (Qwen3.5-9B)."
echo "  Pool location: appworld/${OUT_BASE}/seed_{1..${GROUP_SIZE}}/"
echo "============================================"
