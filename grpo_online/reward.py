"""Online reward: tokenize new trajectories, score segments via a rubric judge,
and compute serpo per-segment advantages per task group, in memory."""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Callable
from grpo.preprocess.make_advantages import RolloutData, compute_serpo_advantage
from grpo.preprocess.tokenize_trajectory import tokenize_trajectory  # patched in tests

MAX_TOKENS = 32000


def score_round(items: list[dict[str, Any]], tokenizer,
                judge_fn: Callable[[str], list[dict]]) -> list[RolloutData]:
    """items: dicts with task_id, seed, lm_calls_path, outcome.
    judge_fn(lm_calls_path) -> list of {contribution, start_step, end_step}."""
    rds: list[RolloutData] = []
    by_task: dict[str, list[RolloutData]] = defaultdict(list)
    for it in items:
        try:
            tok = tokenize_trajectory(it["lm_calls_path"], tokenizer)
        except Exception:
            continue
        if len(tok["input_ids"]) > MAX_TOKENS:
            continue
        segments = judge_fn(it["lm_calls_path"])
        rd = RolloutData(
            task_id=it["task_id"], seed=it["seed"],
            input_ids=tok["input_ids"], attention_mask=tok["attention_mask"],
            response_mask=tok["response_mask"], step_token_ranges=tok["step_token_ranges"],
            segments=segments, outcome=float(it.get("outcome", 0.0)),
        )
        by_task[it["task_id"]].append(rd)
    for group in by_task.values():
        compute_serpo_advantage(group)
        rds.extend(group)
    return rds
