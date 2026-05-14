#!/usr/bin/env bash
# Vanilla GRPO Round-1, FULL pool, CONTINUOUS outcome (pass_rate).
# Stronger baseline: ~84/90 task groups have variance from partial pass rates.
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "[1/3] build vanilla_full_continuous parquet"
python -m grpo.preprocess.build_dataset \
    --method vanilla --condition full --outcome-type continuous \
    --rollout-dir appworld/experiments/outputs/rollout/round0 \
    --joint-dir rubric_reward/results/rollout \
    --output-dir grpo/data/round0

echo "[2/3] train"
python -m grpo.trainer.run_grpo --config grpo/trainer/config/vanilla_full_continuous.yaml

echo "[3/3] eval test_normal (requires vLLM running with --enable-lora)"
python -m grpo.eval.run_test_normal \
    --ckpt grpo/ckpts/vanilla_full_continuous/round1 \
    --method vanilla \
    --output grpo/metrics/vanilla_full_continuous_round1.json

echo "done. metrics: grpo/metrics/vanilla_full_continuous_round1.json"
