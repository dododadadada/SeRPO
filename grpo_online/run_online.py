from __future__ import annotations
import argparse, requests
from grpo_online.config import load_config
from grpo_online.online_loop import run_online, Deps
from grpo_online.online_trainer import OnlineTrainer
from grpo_online.judge import make_api_judge
from grpo_online.rollout import run_rollout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--appworld-bin", required=True)
    ap.add_argument("--experiment", default="rollout/online/qwen35_9b")
    args = ap.parse_args()
    cfg = load_config(args.config)
    trainer = OnlineTrainer(
        model_name=cfg.model_name, lora_target_modules=cfg.lora_target_modules,
        lr=cfg.lr, lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha, device_map="auto",
        kl_beta=cfg.kl_beta, clip_eps=cfg.clip_eps, grad_clip=cfg.grad_clip,
        micro_batch_size=cfg.micro_batch_size, kl_per_token_cap=cfg.kl_per_token_cap,
        outlier_logratio_threshold=cfg.outlier_logratio_threshold,
        max_masked_fraction=cfg.max_masked_fraction)
    # seed adapter so the gen server has something to load on boot
    trainer.save_adapter(cfg.adapter_dir)
    judge = make_api_judge(cfg)
    def reload_servers(path):
        for p in cfg.gen_ports:
            requests.post(f"http://localhost:{p}/reload_adapter", json={"path": path}, timeout=120)
    tasks = [l.strip() for l in open("appworld/data/datasets/train.txt") if l.strip()]
    deps = Deps(all_tasks=tasks,
                rollout=lambda rnd, ts: run_rollout(cfg, rnd, ts, args.appworld_bin, args.experiment),
                judge=judge, tokenizer=trainer.tok, trainer=trainer,
                reload_servers=reload_servers)
    run_online(cfg, deps)

if __name__ == "__main__":
    main()
