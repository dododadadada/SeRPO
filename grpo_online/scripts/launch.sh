#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
CFG=grpo_online/config/online_v1.yaml
source /data/minjeong/.envs/grpo9b/bin/activate
mkdir -p grpo_online/runs/v1
# Seed the adapter dir so the gen server has something to load on boot.
CUDA_VISIBLE_DEVICES=0,1 python -c "from grpo_online.online_trainer import OnlineTrainer; from grpo_online.config import load_config; c=load_config('$CFG'); OnlineTrainer(model_name=c.model_name, lora_target_modules=c.lora_target_modules, lr=c.lr, lora_rank=c.lora_rank, lora_alpha=c.lora_alpha, device_map='auto', kl_beta=c.kl_beta, clip_eps=c.clip_eps, grad_clip=c.grad_clip, micro_batch_size=c.micro_batch_size, kl_per_token_cap=c.kl_per_token_cap, outlier_logratio_threshold=c.outlier_logratio_threshold, max_masked_fraction=c.max_masked_fraction).save_adapter(c.adapter_dir)"
# Gen server (GPU 2):
CUDA_VISIBLE_DEVICES=2 python -m grpo_online.gen_server \
  --base-model Qwen/Qwen3.5-9B --adapter grpo_online/runs/v1/adapter_current \
  --served-name Qwen/Qwen3.5-9B --port 8101 > grpo_online/runs/v1/gen.log 2>&1 &
# Loop (GPU 0,1):
CUDA_VISIBLE_DEVICES=0,1 python -m grpo_online.run_online \
  --config $CFG --appworld-bin /data/minjeong/.conda/envs/appworld/bin/appworld
