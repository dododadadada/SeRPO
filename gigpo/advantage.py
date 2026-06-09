"""GiGPO advantage computation (arXiv 2505.10978). Helpers _to_hashable/_are_similar ported from verl-agent gigpo/core_gigpo.py (Apache-2.0)."""
from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np

from grpo.preprocess.make_advantages import RolloutData

EPS = 1e-8


def _to_hashable(x):
    """Convert an observation into a hashable key for anchor-state grouping.
    Ported from verl-agent gigpo/core_gigpo.py (Apache-2.0)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return tuple(x.flatten().tolist())
    if isinstance(x, (list, tuple)):
        return tuple(_to_hashable(e) for e in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _to_hashable(v)) for k, v in x.items()))
    raise TypeError(f"unhashable observation type: {type(x)}")


def _are_similar(a: str, b: str, threshold: float) -> bool:
    """True if difflib SequenceMatcher ratio >= threshold.
    Ported from verl-agent gigpo/core_gigpo.py (Apache-2.0)."""
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("similarity-based grouping supports only str observations")
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _loo_norm(values: np.ndarray, *, mode: str = "leave_one_out") -> np.ndarray:
    """Normalize a group of scalars by subtracting the group mean.

    mode='leave_one_out' (paper F_norm=1): subtract mean only (a rescaled RLOO;
      the rescale is absorbed into the learning rate).
    mode='std' (paper F_norm=std): also divide by population std.
    A singleton group (len 1) returns 0 (no peers to compare against), matching
    the GiGPO reference's size-1 handling.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size <= 1:
        return np.zeros_like(values)
    centered = values - float(values.mean())
    if mode == "leave_one_out":
        return centered
    if mode == "std":
        return centered / (float(values.std()) + EPS)
    raise ValueError(f"unknown norm mode: {mode!r}")


def _discounted_step_returns(
    num_steps: int, outcome: float, gamma: float,
) -> np.ndarray:
    """Terminal-only discounted return-to-go per step (paper Eq. 5 specialized
    to a single terminal reward). Step k (1-indexed) of an N-step trajectory:
    R_k = gamma^(N-k) * outcome. Returns array of length num_steps (index 0 = step 1).

    NOTE: this is the AppWorld-offline simplification documented in the spec —
    the paper's real step return also carries per-step invalid-action penalties,
    which the frozen binary rollouts do not provide.
    """
    if num_steps <= 0:
        return np.zeros(0, dtype=np.float32)
    exps = np.arange(num_steps - 1, -1, -1, dtype=np.float32)  # [N-1, ..., 1, 0]
    return (gamma ** exps) * float(outcome)


def _build_step_groups(
    anchors_per_rollout: list[list[str]],
    *,
    enable_similarity: bool,
    threshold: float,
) -> list[list[int]]:
    """Anchor-state grouping (paper Eqs. 4, 6) across one task's rollouts.

    Input: anchors_per_rollout[i] = list of anchor strings for rollout i's steps.
    Output: same nested shape, each step replaced by an integer group id; steps
    sharing an anchor (exact, or similarity >= threshold) get the same id.

    Exact mode: hashmap on _to_hashable(anchor). Similarity mode: greedy
    clustering by SequenceMatcher ratio, matching the GiGPO reference.
    Similarity-mode clustering compares each anchor to the first representative
    of each existing cluster, so it is order-dependent (matches the reference).
    """
    if not enable_similarity:
        key_to_id: dict = {}
        out: list[list[int]] = []
        next_id = 0
        for anchors in anchors_per_rollout:
            ids = []
            for a in anchors:
                key = _to_hashable(a)
                if key not in key_to_id:
                    key_to_id[key] = next_id
                    next_id += 1
                ids.append(key_to_id[key])
            out.append(ids)
        return out

    # Similarity mode: greedy representative clustering.
    reps: list[str] = []
    out = []
    for anchors in anchors_per_rollout:
        ids = []
        for a in anchors:
            gid = None
            for j, rep in enumerate(reps):
                if _are_similar(a, rep, threshold):
                    gid = j
                    break
            if gid is None:
                gid = len(reps)
                reps.append(a)
            ids.append(gid)
        out.append(ids)
    return out


def compute_gigpo_advantage(
    rollouts: list[RolloutData],
    *,
    gamma: float = 0.95,
    omega: float = 1.0,
    norm_mode: str = "leave_one_out",
    enable_similarity: bool = False,
    similarity_thresh: float = 0.9,
) -> None:
    """GiGPO advantage (arXiv 2505.10978): A = A^E + omega * A^S.

    A^E: episode-level relative advantage = LOO-normalized binary outcome across
         the group, broadcast uniformly to a rollout's assistant tokens (Eq. 3).
    A^S: step-level relative advantage = LOO-normalized discounted step return
         within each anchor-state group, broadcast to that step's token span
         (Eqs. 4-8). Anchor = the env-output observation preceding the step.

    Operates on one task's group of rollouts; mutates r.token_adv in place.
    """
    # --- A^E: episode advantage (paper Eq. 3) ---
    outcomes = np.asarray([r.outcome for r in rollouts], dtype=np.float32)
    ae = _loo_norm(outcomes, mode=norm_mode)  # one scalar per rollout

    # --- anchor-state grouping (paper Eqs. 4, 6) ---
    anchors_per_rollout = [list(r.step_anchor_obs) for r in rollouts]
    group_ids = _build_step_groups(
        anchors_per_rollout,
        enable_similarity=enable_similarity,
        threshold=similarity_thresh,
    )

    # --- discounted step returns (paper Eq. 5, terminal-only) ---
    step_returns = [
        _discounted_step_returns(len(r.step_token_ranges), r.outcome, gamma)
        for r in rollouts
    ]

    # --- pool step returns by group id, LOO-normalize within group (Eq. 7) ---
    gid_to_returns: dict[int, list[float]] = {}
    for i, ids in enumerate(group_ids):
        for k, gid in enumerate(ids):
            gid_to_returns.setdefault(gid, []).append(float(step_returns[i][k]))
    gid_to_mean: dict[int, float] = {}
    gid_to_std: dict[int, float] = {}
    for gid, vals in gid_to_returns.items():
        arr = np.asarray(vals, dtype=np.float32)
        gid_to_mean[gid] = float(arr.mean())
        if norm_mode == "std":
            gid_to_std[gid] = float(arr.std())

    # --- write A = A^E + omega * A^S onto tokens ---
    for i, r in enumerate(rollouts):
        r.token_adv = np.zeros(len(r.input_ids), dtype=np.float32)
        mask = np.asarray(r.response_mask, dtype=bool)
        r.token_adv[mask] = ae[i]
        ids = group_ids[i]
        for k, (tok_start, tok_end) in enumerate(r.step_token_ranges):
            gid = ids[k]
            n_in_group = len(gid_to_returns[gid])
            if n_in_group <= 1:
                a_s = 0.0  # singleton group -> no relative signal
            else:
                centered = float(step_returns[i][k]) - gid_to_mean[gid]
                if norm_mode == "std":
                    a_s = centered / (gid_to_std[gid] + EPS)
                else:  # leave_one_out
                    a_s = centered
            r.token_adv[tok_start:tok_end] += omega * a_s
