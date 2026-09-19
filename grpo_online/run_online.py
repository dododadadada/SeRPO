from __future__ import annotations
import argparse, functools, logging, os, requests
from grpo_online.config import load_config
from grpo_online.online_loop import run_online, Deps
from grpo_online.online_trainer import OnlineTrainer
from grpo_online.judge import make_api_judge
from grpo_online.rollout import run_rollout
from grpo_online.reward import score_round
from grpo_online.vllm_gen import VllmGenServer

def _dry_judge(lm_calls_path):
    """Stub judge for --dry-judge: one segment over all steps, contribution 5..1
    decreasing with the number of LM calls, so advantages are non-degenerate."""
    n_steps = sum(1 for l in open(lm_calls_path) if l.strip())
    return [{"start_step": 1, "end_step": n_steps, "contribution": float(max(1, 5 - n_steps // 10))}]


def main():
    # Configure logging so the "online" logger's per-round metrics (loss, kl,
    # guard skips, dropped items) are actually emitted — without this the whole
    # run is unobservable (logger.info is silently dropped by the root WARNING).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--appworld-bin", required=True)
    ap.add_argument("--experiment", default="rollout/online/qwen35_9b")
    ap.add_argument("--dry-judge", action="store_true",
                    help="plumbing smoke without the paid rubric API: a stub judge "
                         "scores each trajectory as one segment whose contribution "
                         "falls with its step count (no OPENAI_API_KEY needed).")
    ap.add_argument("--resume-round", type=int, default=0,
                    help="resume training: load the existing adapter_dir (weights "
                         "from a prior run) and start the loop at this round number. "
                         "0 = fresh start (seed a base adapter at round 1).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    resume = args.resume_round > 0
    trainer = OnlineTrainer(
        model_name=cfg.model_name, lora_target_modules=cfg.lora_target_modules,
        lr=cfg.lr, lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha, device_map="auto",
        kl_beta=cfg.kl_beta, clip_eps=cfg.clip_eps, grad_clip=cfg.grad_clip,
        micro_batch_size=cfg.micro_batch_size, kl_per_token_cap=cfg.kl_per_token_cap,
        outlier_logratio_threshold=cfg.outlier_logratio_threshold,
        max_masked_fraction=cfg.max_masked_fraction,
        ratio_halt_threshold=cfg.ratio_halt_threshold,
        resume_adapter=cfg.adapter_dir if resume else None)
    # Fresh start: seed a base adapter so the gen server has something to load on
    # boot. Resume: adapter_dir already holds the prior weights — leave them.
    if not resume:
        trainer.save_adapter(cfg.adapter_dir)
    judge = _dry_judge if args.dry_judge else make_api_judge(cfg)
    tasks = [l.strip() for l in open("appworld/data/datasets/train.txt") if l.strip()]

    server = None
    if cfg.gen_backend == "vllm":
        # run_online owns the vLLM server: base loaded once, LoRA hot-swapped per
        # round. Started after the trainer so the trainer's GPU footprint is
        # already committed (the two live on different GPUs anyway).
        server = VllmGenServer(cfg, port=cfg.gen_ports[0], gpus=",".join(cfg.gen_gpus),
                               log_path=os.path.join(cfg.output_dir, "vllm.log"))
        server.start()
        server.load_adapter(cfg.adapter_dir)
        reload_servers = server.load_adapter
    else:
        def reload_servers(path):
            for p in cfg.gen_ports:
                requests.post(f"http://localhost:{p}/reload_adapter", json={"path": path}, timeout=120)

    deps = Deps(all_tasks=tasks,
                rollout=lambda rnd, ts: run_rollout(cfg, rnd, ts, args.appworld_bin, args.experiment),
                judge=judge, tokenizer=trainer.tok, trainer=trainer,
                reload_servers=reload_servers,
                score_round=functools.partial(score_round, max_workers=cfg.judge_workers))
    try:
        run_online(cfg, deps, start_round=args.resume_round if resume else 1)
    finally:
        if server is not None:
            server.stop()

if __name__ == "__main__":
    main()
