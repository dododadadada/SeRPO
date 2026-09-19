#!/usr/bin/env bash
# In-run validation: whenever the training loop saves a checkpoint whose round is a
# multiple of $EVERY, evaluate it on the dev split against the loop's OWN vLLM server,
# using the second LoRA slot (--max-loras 2) so the loop's adapter is never touched.
#
# The eval mostly overlaps the loop's training phase (GPU 1 idle for ~80% of a round),
# so it costs little wall-clock; it uses no extra GPU and no judge API calls.
#
# Usage: bash grpo_online/scripts/val_during_run.sh [config] [every_n_rounds] [dataset]
set -uo pipefail
cd "$(dirname "$0")/../.."
CFG="${1:-grpo_online/config/online_v2_lr1e5.yaml}"; EVERY="${2:-4}"; DS="${3:-dev}"
read -r OUT PORT < <(python3 -c "
import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print(c['output_dir'], c['gen_ports'][0])" "$CFG")
# eval_ckpts writes to <output_dir>/eval/<out-name>; keep the watcher's bookkeeping
# in that same directory so the "already evaluated" check actually sees summary.json.
VAL="$OUT/eval/val_$DS"; mkdir -p "$VAL"
echo "[$(date)] in-run validation: cfg=$CFG every=$EVERY rounds, dataset=$DS, server port=$PORT -> $VAL"
while true; do
  # Stop when the training loop is gone (the post-run full eval takes over).
  pgrep -f "run_online --config $CFG" > /dev/null || { echo "[$(date)] training loop gone; validator exiting"; break; }
  for CK in $(ls -d "$OUT"/ckpt_round_* 2>/dev/null | grep -vE '_vllm$' | sort -t_ -k3 -n); do
    N="${CK##*_}"
    (( N % EVERY == 0 )) || continue
    grep -q "\"$(basename "$CK")\"" "$VAL/summary.json" 2>/dev/null && continue
    echo "[$(date)] evaluating $(basename "$CK") on $DS"
    HF_HUB_OFFLINE=1 .venv/bin/python -m grpo_online.eval_ckpts --config "$CFG" --gpu 1 \
      --attach-port "$PORT" --lora-name eval_probe --dataset "$DS" --out-name "val_$DS" \
      --num-processes 4 --ckpts "$CK" >> "$VAL/val.log" 2>&1
  done
  sleep 300
done
