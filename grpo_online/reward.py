"""Online reward: tokenize new trajectories, score segments via a rubric judge,
and compute serpo per-segment advantages per task group, in memory."""
from __future__ import annotations
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
from grpo.preprocess.make_advantages import RolloutData, compute_serpo_advantage
from grpo.preprocess.tokenize_trajectory import tokenize_trajectory  # patched in tests

logger = logging.getLogger("online")

MAX_TOKENS = 32000


def score_round(items: list[dict[str, Any]], tokenizer,
                judge_fn: Callable[[str], list[dict]],
                max_workers: int = 1) -> list[RolloutData]:
    """items: dicts with task_id, seed, lm_calls_path, outcome.
    judge_fn(lm_calls_path) -> list of {contribution, start_step, end_step}.
    max_workers > 1 runs the judge calls (remote API) concurrently; only items
    that passed tokenization/length are submitted, and results are consumed in
    item order so the output matches the sequential path."""
    rds: list[RolloutData] = []
    by_task: dict[str, list[RolloutData]] = defaultdict(list)
    dropped = {"tokenize": 0, "too_long": 0, "judge": 0}
    # Pass 1 (local, cheap): tokenize + length filter.
    kept: list[tuple[dict, dict]] = []
    for it in items:
        path = it.get("lm_calls_path")
        try:
            tok = tokenize_trajectory(path, tokenizer)
        except Exception as e:
            dropped["tokenize"] += 1
            logger.warning("dropping %s seed=%s: tokenize failed: %s",
                           it.get("task_id"), it.get("seed"), e)
            continue
        if len(tok["input_ids"]) > MAX_TOKENS:
            dropped["too_long"] += 1
            logger.warning("dropping %s seed=%s: %d tokens > MAX_TOKENS=%d",
                           it.get("task_id"), it.get("seed"),
                           len(tok["input_ids"]), MAX_TOKENS)
            continue
        kept.append((it, tok))
    # Pass 2: judge. Can be a remote API; a single failure must not abort the round.
    if max_workers > 1 and len(kept) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(judge_fn, it["lm_calls_path"]) for it, _ in kept]
            results = []
            for f in futs:
                try:
                    results.append(f.result())
                except Exception as e:  # keep the exception, decide below
                    results.append(e)
    else:
        results = []
        for it, _ in kept:
            try:
                results.append(judge_fn(it["lm_calls_path"]))
            except Exception as e:
                results.append(e)
    for (it, tok), segments in zip(kept, results):
        if isinstance(segments, Exception):
            dropped["judge"] += 1
            logger.warning("dropping %s seed=%s: judge failed: %s",
                           it.get("task_id"), it.get("seed"), segments)
            continue
        rd = RolloutData(
            task_id=it["task_id"], seed=it["seed"],
            input_ids=tok["input_ids"], attention_mask=tok["attention_mask"],
            response_mask=tok["response_mask"], step_token_ranges=tok["step_token_ranges"],
            segments=segments, outcome=float(it.get("outcome", 0.0)),
        )
        by_task[it["task_id"]].append(rd)
    # Empty task-groups are never created (only successfully-scored items are
    # appended), so compute_serpo_advantage always gets a non-empty group.
    for group in by_task.values():
        compute_serpo_advantage(group)
        rds.extend(group)
    total_dropped = sum(dropped.values())
    if total_dropped:
        logger.warning(
            "score_round: dropped %d/%d items (tokenize=%d too_long=%d judge=%d);"
            " %d scored",
            total_dropped, len(items), dropped["tokenize"],
            dropped["too_long"], dropped["judge"], len(rds))
    return rds
