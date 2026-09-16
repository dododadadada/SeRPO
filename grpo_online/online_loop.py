from __future__ import annotations
import logging, os, random, shutil
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("online")

@dataclass
class Deps:
    all_tasks: list
    rollout: Callable[[int, list], list]
    judge: Callable[[str], list]
    tokenizer: Any
    trainer: Any
    reload_servers: Callable[[str], None]
    score_round: Callable = None

def run_online(cfg, deps: Deps, start_round: int = 1) -> int:
    if deps.score_round is None:
        from grpo_online.reward import score_round as _sr
        deps.score_round = _sr
    # Offset the seed by the resume point so a resumed run doesn't replay the
    # exact task draws of the rounds it already trained on (start_round=1 -> seed).
    rng = random.Random(cfg.seed + start_round - 1)
    from grpo_online.rollout import sample_tasks
    done = start_round - 1
    for rnd in range(start_round, cfg.N_rounds + 1):
        tasks = sample_tasks(deps.all_tasks, cfg.M, rng)
        logger.info("round %d/%d: rolling out %d task(s): %s",
                    rnd, cfg.N_rounds, len(tasks), tasks)
        items = deps.rollout(rnd, tasks)
        outcomes = [it.get("outcome") for it in items] if items else []
        logger.info("round %d: %d rollouts collected, outcomes=%s",
                    rnd, len(outcomes), outcomes)
        rds = deps.score_round(items, deps.tokenizer, deps.judge)
        if not rds:
            logger.warning(
                "round %d: no scored rollouts (items=%d) — skipping round",
                rnd, len(items) if items is not None else 0)
            done = rnd
            continue
        metrics = deps.trainer.train_on_batch(rds, cfg.K)
        done = rnd
        logger.info(
            "round %d: loss=%.4f kl=%.4f ratio_max=%.3f masked=%d/%d skipped=%s (rds=%d)",
            rnd, metrics.get("loss", 0.0), metrics.get("kl_loss", 0.0),
            metrics.get("ratio_max", 0.0), metrics.get("masked_tokens", 0),
            metrics.get("resp_tokens", 0), metrics.get("step_skipped"), len(rds))
        if metrics.get("kl_loss", 0.0) > cfg.kl_halt_threshold:
            logger.error("round %d kl=%.3f exceeded halt — stopping", rnd, metrics["kl_loss"])
            break
        deps.trainer.save_adapter(cfg.adapter_dir)
        ckpt_every = getattr(cfg, "ckpt_every", 0)
        if ckpt_every and rnd % ckpt_every == 0:
            snap = os.path.join(cfg.output_dir, f"ckpt_round_{rnd}")
            if os.path.exists(snap):
                shutil.rmtree(snap)
            shutil.copytree(cfg.adapter_dir, snap)
            logger.info("round %d: checkpoint snapshot -> %s", rnd, snap)
        deps.reload_servers(cfg.adapter_dir)
    return done
