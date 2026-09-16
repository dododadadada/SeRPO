#!/usr/bin/env bash
# Serve one VANILLA v3i checkpoint via peft_chat_server on a dedicated GPU/port,
# run the AppWorld dev eval, then stop the server. Mirrors eval_one_ckpt.sh
# (which is for SeRPO) so SeRPO-vs-vanilla numbers come from an identical path.
# Usage: eval_vanilla_ckpt.sh <STEP|final> <GPU> <PORT>
set -euo pipefail
STEP="$1"; GPU="$2"; PORT="$3"
ROOT=/data/minjeong/Autonomous_agent
cd "$ROOT"

if [ "$STEP" = "final" ]; then
  LORA="grpo/ckpts/serpo_avg_9b_v3i/round1"
  NAME="serpoavg_9b_v3i_step76"; CFG="serpoavg_9b_v3i_step_76"
else
  LORA="grpo/ckpts/serpo_avg_9b_v3i/round1/step_${STEP}"
  NAME="serpoavg_9b_v3i_step${STEP}"; CFG="serpoavg_9b_v3i_step_${STEP}"
fi
CKPT_LABEL="${LORA#grpo/ckpts/}"

# Build eval config from the serpo step_40 template (same non-thinking/temp0).
TPL="appworld/experiments/configs/eval/dev/qwen35_9b_v3i_step_40.jsonnet"
OUT="appworld/experiments/configs/eval/dev/${CFG}.jsonnet"
sed -e "s/qwen35_9b_v3i_step40/${NAME}/g" \
    -e "s#serpo_9b_full_continuous_v3i/round1/step_40#${CKPT_LABEL}#g" \
    "$TPL" > "$OUT"

source /data/minjeong/.envs/grpo9b/bin/activate
CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python -m grpo.eval.peft_chat_server \
  --base-model Qwen/Qwen3.5-9B --lora "$LORA" --served-name "$NAME" --port "$PORT" --device-map cuda:0 \
  > "grpo/eval/logs/serve_${CFG}.log" 2>&1 &
SRV=$!
for _ in $(seq 1 120); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
  ps -p "$SRV" >/dev/null 2>&1 || { echo "SERVER DIED step=$STEP"; exit 1; }
  sleep 5
done

cd "$ROOT/appworld"
VLLM_PORT="$PORT" NO_API_KEY=dummy OPENAI_API_KEY=dummy \
  /data/minjeong/.conda/envs/appworld/bin/appworld run "eval/dev/${CFG}" \
  --num-processes 4 --with-evaluation --without-setup \
  > "$ROOT/grpo/eval/logs/eval_${CFG}.log" 2>&1 || true

kill "$SRV" 2>/dev/null || true
echo "=== DONE vanilla step=$STEP ==="
grep -A6 "Text Evaluation Report" "$ROOT/grpo/eval/logs/eval_${CFG}.log" 2>/dev/null | tail -8
