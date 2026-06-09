"""Compute per-token advantage tensors for Vanilla GRPO and SeRPO (GiGPO
advantage lives in gigpo/advantage.py).

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
    outcome: float  # ∈ [0, 1]; binary {0.0, 1.0} or continuous pass_rate
    # Per-step anchor observation (env-output text preceding each assistant
    # step). Used only by GiGPO; empty for other methods. Length == num_steps.
    step_anchor_obs: list[str] = field(default_factory=list)
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


def compute_serpo_avg_advantage(rollouts: list[RolloutData]) -> None:
    """Ablation isolating SeRPO's segment-level PLACEMENT.

    Same rubric contributions as serpo, but collapsed to ONE trajectory reward =
    token-weighted mean of segment contributions (weight = #tokens in the segment
    span). Then z-score these trajectory rewards across the group and broadcast
    uniformly to assistant tokens. Identical reward source + per-token raw total
    as serpo; only the granularity (per-segment -> trajectory) and the z-score
    level differ — exactly the segment-vs-trajectory contrast being ablated.
    """
    traj: list[float] = []
    for r in rollouts:
        num = 0.0
        den = 0
        for seg in r.segments:
            seg_len = 0
            for k in range(int(seg["start_step"]), int(seg["end_step"]) + 1):
                idx = k - 1
                if idx < 0 or idx >= len(r.step_token_ranges):
                    continue
                tok_start, tok_end = r.step_token_ranges[idx]
                seg_len += tok_end - tok_start
            num += float(seg["contribution"]) * seg_len
            den += seg_len
        traj.append(num / den if den > 0 else 0.0)
    arr = np.asarray(traj, dtype=np.float32)
    mu = float(arr.mean())
    sigma = float(arr.std()) + EPS
    for r, tr in zip(rollouts, traj):
        scalar = (tr - mu) / sigma
        mask = np.asarray(r.response_mask, dtype=bool)
        r.token_adv = np.zeros(len(r.input_ids), dtype=np.float32)
        r.token_adv[mask] = scalar
