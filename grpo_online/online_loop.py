from __future__ import annotations
import logging, random
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

def run_online(cfg, deps: Deps) -> int:
    if deps.score_round is None:
        from grpo_online.reward import score_round as _sr
        deps.score_round = _sr
    rng = random.Random(cfg.seed)
    from grpo_online.rollout import sample_tasks
    done = 0
    for rnd in range(1, cfg.N_rounds + 1):
        tasks = sample_tasks(deps.all_tasks, cfg.M, rng)
        items = deps.rollout(rnd, tasks)
        rds = deps.score_round(items, deps.tokenizer, deps.judge)
        metrics = deps.trainer.train_on_batch(rds, cfg.K)
        done = rnd
        logger.info("round %d: loss=%.4f kl=%.4f (rds=%d)",
                    rnd, metrics.get("loss", 0.0), metrics.get("kl_loss", 0.0), len(rds))
        if metrics.get("kl_loss", 0.0) > cfg.kl_halt_threshold:
            logger.error("round %d kl=%.3f exceeded halt — stopping", rnd, metrics["kl_loss"])
            break
        deps.trainer.save_adapter(cfg.adapter_dir)
        deps.reload_servers(cfg.adapter_dir)
    return done
