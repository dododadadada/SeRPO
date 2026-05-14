"""Compute per-token advantage tensors for Vanilla GRPO and SeRPO.

Both operate on a per-task group of 8 rollouts. The advantage is broadcast
onto each rollout's pre-existing token layout (assistant token spans).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

EPS = 1e-8


@dataclass
class RolloutData:
    """One rollout's preprocessed view, mutated in-place by advantage funcs."""

    task_id: str
    seed: int
    input_ids: list[int]
    attention_mask: list[int]
    response_mask: list[int]
    step_token_ranges: list[tuple[int, int]]
    segments: list[dict[str, Any]]
    outcome: int
    token_adv: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.token_adv is None:
            self.token_adv = np.zeros(len(self.input_ids), dtype=np.float32)


def compute_vanilla_advantage(rollouts: list[RolloutData]) -> None:
    """Outcome-only z-score, broadcast trajectory-uniformly to assistant tokens."""
    outcomes = np.asarray([r.outcome for r in rollouts], dtype=np.float32)
    mu = float(outcomes.mean())
    sigma = float(outcomes.std()) + EPS
    for r in rollouts:
        scalar = (float(r.outcome) - mu) / sigma
        mask = np.asarray(r.response_mask, dtype=bool)
        r.token_adv = np.zeros(len(r.input_ids), dtype=np.float32)
        r.token_adv[mask] = scalar


def compute_serpo_advantage(rollouts: list[RolloutData]) -> None:
    """Per-segment z-score over pooled contributions of all rollouts in the
    group; broadcast each segment's z-score to its assistant token span."""
    pool = np.asarray(
        [seg["contribution"] for r in rollouts for seg in r.segments],
        dtype=np.float32,
    )
    if pool.size == 0:
        for r in rollouts:
            r.token_adv = np.zeros(len(r.input_ids), dtype=np.float32)
        return
    mu = float(pool.mean())
    sigma = float(pool.std()) + EPS
    for r in rollouts:
        r.token_adv = np.zeros(len(r.input_ids), dtype=np.float32)
        for seg in r.segments:
            a_hat = (float(seg["contribution"]) - mu) / sigma
            start_step = int(seg["start_step"])
            end_step = int(seg["end_step"])
            for k in range(start_step, end_step + 1):
                idx = k - 1
                if idx < 0 or idx >= len(r.step_token_ranges):
                    continue  # segment refers to nonexistent step
                tok_start, tok_end = r.step_token_ranges[idx]
                r.token_adv[tok_start:tok_end] = a_hat
