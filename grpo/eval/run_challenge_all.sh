#!/usr/bin/env bash
# Evaluate all 5 methods on the AppWorld test_challenge set, reusing the exact
# checkpoint steps from the test_normal comparison so numbers are comparable:
#   base (no LoRA), binary=step60, continuous=step60, gigpo=step70, serpo=step50.
# 4 GPUs (0-3) for 5 jobs: launch 4 in parallel, then start serpo on the first
# freed GPU. Each method is a self-contained serve+eval via eval_test_ckpt.sh.
set -u
ROOT=/data/minjeong/Autonomous_agent
E="$ROOT/grpo/eval/eval_test_ckpt.sh"
DS=test_challenge
LOGD="$ROOT/grpo/eval/logs"

declare -A G   # pid -> gpu
launch() {  # name lora gpu port
  bash "$E" "$1" "$2" "$3" "$4" "$DS" > "$LOGD/run_$1.out" 2>&1 &
  G[$!]=$3
  echo "launched $1 on GPU $3 port $4 (pid $!)"
}

launch base_chal       NONE 0 8120
launch binary60_chal   grpo/ckpts/vanilla_9b_full_binary_v3i/round1/step_60     1 8121
launch continuous60_chal grpo/ckpts/vanilla_9b_full_continuous_v3i/round1/step_60 2 8122
launch gigpo70_chal    grpo/ckpts/gigpo_9b_full_binary_v3i/round1/step_70       3 8123

# When the first of the four finishes, reuse its GPU for serpo.
serpo_done=0
while [ $serpo_done -eq 0 ]; do
  for pid in "${!G[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      gpu=${G[$pid]}
      unset 'G[$pid]'
      echo "GPU $gpu freed (pid $pid done) -> launching serpo there"
      launch serpo50_chal grpo/ckpts/serpo_9b_full_continuous_v3i/round1/step_50 "$gpu" 8124
      serpo_done=1
      break
    fi
  done
  [ $serpo_done -eq 0 ] && sleep 30
done

wait
echo "===== ALL CHALLENGE EVALS DONE ====="
for f in base_chal binary60_chal continuous60_chal gigpo70_chal serpo50_chal; do
  echo "----- $f -----"
  grep -A6 "Text Evaluation Report" "$LOGD/eval_${f}_${DS}.log" 2>/dev/null | tail -7
done
