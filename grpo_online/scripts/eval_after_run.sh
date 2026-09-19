#!/usr/bin/env bash
# Wait for the online run to finish, then evaluate base + all checkpoints on dev,
# sharded across two GPUs (two vLLM servers), and merge the two summaries.
# Usage: bash grpo_online/scripts/eval_after_run.sh <run_online_pid> [config] [dataset] [gpuA] [gpuB]
set -uo pipefail
cd "$(dirname "$0")/../.."
PID="$1"; CFG="${2:-grpo_online/config/online_v1_vllm.yaml}"; DS="${3:-dev}"; GA="${4:-0}"; GB="${5:-1}"
OUT=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CFG")
echo "[$(date)] waiting for run_online pid $PID to exit"
while kill -0 "$PID" 2>/dev/null; do sleep 120; done
echo "[$(date)] run_online exited; rounds trained: $(grep -c 'loss=' "$OUT/loop.log")"
sleep 30  # let vLLM release GPU memory
mkdir -p appworld/experiments/configs && cp -r grpo_online/appworld_configs/. appworld/experiments/configs/
python3 grpo_online/appworld_patches.py
# Shard: even-indexed checkpoints (+ base) -> GPU A, odd-indexed -> GPU B.
mapfile -t CK < <(ls -d "$OUT"/ckpt_round_* 2>/dev/null | sort -t_ -k3 -n)
A=(base); B=()
for i in "${!CK[@]}"; do if (( i % 2 == 0 )); then A+=("${CK[$i]}"); else B+=("${CK[$i]}"); fi; done
echo "[$(date)] GPU $GA: ${#A[@]} targets; GPU $GB: ${#B[@]} targets"
mkdir -p "$OUT/eval"
HF_HUB_OFFLINE=1 .venv/bin/python -m grpo_online.eval_ckpts --config "$CFG" --gpu "$GA" --port 8102 --dataset "$DS" --out-name "${DS}_a" --ckpts "${A[@]}" > "$OUT/eval/${DS}_a.log" 2>&1 &
PA=$!
HF_HUB_OFFLINE=1 .venv/bin/python -m grpo_online.eval_ckpts --config "$CFG" --gpu "$GB" --port 8103 --dataset "$DS" --out-name "${DS}_b" --ckpts "${B[@]}" > "$OUT/eval/${DS}_b.log" 2>&1 &
PB=$!
wait $PA; wait $PB
echo "[$(date)] both shards done; merging"
.venv/bin/python - "$OUT/eval" "$DS" <<'PY'
import json, re, sys
from pathlib import Path
root, ds = Path(sys.argv[1]), sys.argv[2]
rows = []
for sub in (f"{ds}_a", f"{ds}_b"):
    f = root / sub / "summary.json"
    if f.exists(): rows += json.loads(f.read_text())
key = lambda r: (r["name"] != "base", int(re.search(r"(\d+)", r["name"]).group(1)) if r["name"] != "base" else 0)
rows.sort(key=key)
(root / f"{ds}_summary.json").write_text(json.dumps(rows, indent=1))
md = f"| checkpoint | TGC | SGC | mean test-pass | n |\n|---|---|---|---|---|\n" + "\n".join(
    f"| {r['name']} | {r['tgc']:.3f} | {r['sgc']:.3f} | {r['mean_test_pass_frac']:.3f} | {r['n_tasks']} |" for r in rows) + "\n"
(root / f"{ds}_summary.md").write_text(md); print(md)
PY
echo "[$(date)] EVAL DONE -> $OUT/eval/${DS}_summary.md"
