#!/usr/bin/env bash
# Serve one checkpoint (or base) via peft_chat_server and run an AppWorld eval on
# a chosen dataset (default test_normal). Same path as the dev evals → comparable.
# Usage: eval_test_ckpt.sh <NAME> <LORA_DIR|NONE> <GPU> <PORT> [DATASET] [NUM_PROCESSES]
set -euo pipefail
NAME="$1"; LORA="$2"; GPU="$3"; PORT="$4"; DS="${5:-test_normal}"; NP="${6:-4}"
ROOT=/data/minjeong/Autonomous_agent; cd "$ROOT"
CFG="${NAME}_${DS}"

TPL="appworld/experiments/configs/eval/dev/qwen35_9b_v3i_step_40.jsonnet"
OUT="appworld/experiments/configs/eval/dev/${CFG}.jsonnet"
sed -e "s/qwen35_9b_v3i_step40/${NAME}/g" \
    -e 's/"dataset": "dev"/"dataset": "'"${DS}"'"/' \
    "$TPL" > "$OUT"

source /data/minjeong/.envs/grpo9b/bin/activate
if [ "$LORA" = "NONE" ]; then LARG=""; else LARG="--lora $LORA"; fi
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python -m grpo.eval.peft_chat_server \
  --base-model Qwen/Qwen3.5-9B $LARG --served-name "$NAME" --port "$PORT" --device-map cuda:0 \
  > "grpo/eval/logs/serve_${CFG}.log" 2>&1 &
SRV=$!
for _ in $(seq 1 120); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
  ps -p "$SRV" >/dev/null 2>&1 || { echo "SERVER DIED $NAME"; exit 1; }
  sleep 5
done

cd "$ROOT/appworld"
VLLM_PORT="$PORT" NO_API_KEY=dummy OPENAI_API_KEY=dummy \
  /data/minjeong/.conda/envs/appworld/bin/appworld run "eval/dev/${CFG}" \
  --num-processes "$NP" --with-evaluation --without-setup \
  > "$ROOT/grpo/eval/logs/eval_${CFG}.log" 2>&1 || true
kill "$SRV" 2>/dev/null || true
echo "=== DONE $NAME on $DS ==="
grep -A6 "Text Evaluation Report" "$ROOT/grpo/eval/logs/eval_${CFG}.log" 2>/dev/null | tail -8
