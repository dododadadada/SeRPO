#!/usr/bin/env bash
# End-to-end Vanilla GRPO Round-1 run: preprocess → train → eval.
#
# Prerequisites:
#   - conda env "appworld" active
#   - external vLLM running with --enable-lora on http://localhost:8000
#   - Phase 2 joint reward output present at rubric_reward/results/rollout/joint_seed_N.jsonl
#     (not strictly required for vanilla, but the build_dataset orchestrator
#     loads it consistently for both methods)
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "[1/3] build vanilla parquet"
python -m grpo.preprocess.build_dataset \
    --method vanilla --condition failonly \
    --rollout-dir appworld/experiments/outputs/rollout/round0 \
    --joint-dir rubric_reward/results/rollout \
    --output-dir grpo/data/round0

echo "[2/3] train"
python -m grpo.trainer.run_grpo --config grpo/trainer/config/vanilla_round1.yaml

echo "[3/3] eval test_normal (requires vLLM running with --enable-lora)"
python -m grpo.eval.run_test_normal \
    --ckpt grpo/ckpts/vanilla/round1 \
    --method vanilla \
    --output grpo/metrics/vanilla_round1.json

echo "done. metrics: grpo/metrics/vanilla_round1.json"
