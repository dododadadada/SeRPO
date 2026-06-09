"""Tests for grpo.preprocess.make_advantages."""
from __future__ import annotations

import numpy as np

from grpo.preprocess.make_advantages import (
    RolloutData,
    _are_similar,
    _build_step_groups,
    _discounted_step_returns,
    _loo_norm,
    _to_hashable,
    compute_serpo_advantage,
    compute_vanilla_advantage,
)


def _mock_rollout(
    num_tokens: int,
    response_mask: list[int],
    step_token_ranges: list[tuple[int, int]],
    segments: list[dict],
    outcome: int,
) -> RolloutData:
    return RolloutData(
        task_id="t",
        seed=1,
        input_ids=list(range(num_tokens)),
        attention_mask=[1] * num_tokens,
        response_mask=list(response_mask),
        step_token_ranges=list(step_token_ranges),
        segments=list(segments),
        outcome=outcome,
    )


def test_vanilla_all_zero_outcomes_yields_zero_advantage():
    """In fail-only set, all 8 outcomes = 0 → advantage scalar = 0 → all-zero tensor."""
    rollouts = [
        _mock_rollout(
            10,
            [0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
            [(2, 5), (6, 8)],
            [],
            0,
        )
        for _ in range(8)
    ]
    compute_vanilla_advantage(rollouts)
    for r in rollouts:
        assert np.allclose(r.token_adv, 0.0)


def test_vanilla_mixed_outcomes_has_signal():
    """Mixed outcomes → nonzero advantage on assistant tokens, zero on others."""
    rollouts = []
    for i in range(8):
        outcome = 1 if i < 2 else 0  # 2 successes
        rollouts.append(
            _mock_rollout(
                10,
                [0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
                [(2, 5), (6, 8)],
                [],
                outcome,
            )
        )
    compute_vanilla_advantage(rollouts)
    # response_mask=0 positions stay zero
    for r in rollouts:
        for i, m in enumerate(r.response_mask):
            if m == 0:
                assert r.token_adv[i] == 0.0
    # Advantage on assistant tokens is the rollout-level scalar; identical across tokens
    for r in rollouts:
        active = [r.token_adv[i] for i, m in enumerate(r.response_mask) if m == 1]
        assert all(v == active[0] for v in active)
    # Group mean of trajectory-level scalars ≈ 0
    scalars = [
        next(r.token_adv[i] for i, m in enumerate(r.response_mask) if m == 1)
        for r in rollouts
    ]
    assert abs(float(np.mean(scalars))) < 1e-5


def test_serpo_segment_broadcast():
    """SeRPO: each segment's tokens get the same z-scored contribution."""
    rollouts = []
    for i in range(8):
        contrib = (i % 5) + 1  # 1..5 distributed
        segs = [
            {"start_step": 1, "end_step": 1, "contribution": contrib},
            {"start_step": 2, "end_step": 2, "contribution": 6 - contrib},
        ]
        rollouts.append(
            _mock_rollout(
                10,
                [0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
                [(2, 5), (6, 8)],
                segs,
                0,
            )
        )
    compute_serpo_advantage(rollouts)
    # Each segment's tokens have identical advantage
    for r in rollouts:
        for seg in r.segments:
            tok_start, tok_end = r.step_token_ranges[seg["start_step"] - 1]
            vals = r.token_adv[tok_start:tok_end]
            assert np.allclose(vals, vals[0])
    # The z-score guarantees mean ≈ 0 over the pooled SEGMENTS (not tokens —
    # tokens get unequal weight when segment lengths differ).
    seg_advs = []
    for r in rollouts:
        for seg in r.segments:
            tok_start, _ = r.step_token_ranges[seg["start_step"] - 1]
            seg_advs.append(float(r.token_adv[tok_start]))
    assert abs(float(np.mean(seg_advs))) < 1e-5


def test_serpo_zero_outside_assistant():
    """Tokens with response_mask=0 stay at zero."""
    rollouts = []
    for i in range(8):
        segs = [{"start_step": 1, "end_step": 1, "contribution": (i % 5) + 1}]
        rollouts.append(
            _mock_rollout(
                10,
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [(2, 5)],
                segs,
                0,
            )
        )
    compute_serpo_advantage(rollouts)
    for r in rollouts:
        for i, m in enumerate(r.response_mask):
            if m == 0:
                assert r.token_adv[i] == 0.0


def test_serpo_steps_in_segmentation_gap_stay_zero():
    """Steps not covered by any segment → token_adv = 0."""
    rollouts = []
    for i in range(8):
        segs = [{"start_step": 1, "end_step": 1, "contribution": (i % 5) + 1}]
        rollouts.append(
            _mock_rollout(
                10,
                [0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
                [(2, 5), (6, 8)],
                segs,
                0,
            )
        )
    compute_serpo_advantage(rollouts)
    # Step 2 tokens [6:8] should remain 0
    for r in rollouts:
        assert np.all(r.token_adv[6:8] == 0.0)


def test_serpo_segment_spanning_multiple_steps():
    """A segment with end_step > start_step covers all steps in the inclusive range."""
    rollouts = []
    for i in range(8):
        segs = [{"start_step": 1, "end_step": 2, "contribution": (i % 5) + 1}]
        rollouts.append(
            _mock_rollout(
                10,
                [0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
                [(2, 5), (6, 8)],
                segs,
                0,
            )
        )
    compute_serpo_advantage(rollouts)
    for r in rollouts:
        v_step1 = r.token_adv[2]
        v_step2 = r.token_adv[6]
        assert v_step1 == v_step2  # same segment → same advantage


def test_rollout_data_has_step_anchor_obs_default():
    """RolloutData accepts an optional step_anchor_obs, defaulting to empty list."""
    r = RolloutData(
        task_id="t", seed=1,
        input_ids=[0, 1, 2], attention_mask=[1, 1, 1],
        response_mask=[0, 1, 1], step_token_ranges=[(1, 3)],
        segments=[], outcome=0.0,
    )
    assert r.step_anchor_obs == []
    r2 = RolloutData(
        task_id="t", seed=1,
        input_ids=[0, 1, 2], attention_mask=[1, 1, 1],
        response_mask=[0, 1, 1], step_token_ranges=[(1, 3)],
        segments=[], outcome=0.0, step_anchor_obs=["obs1"],
    )
    assert r2.step_anchor_obs == ["obs1"]


def test_loo_norm_subtracts_mean_only():
    """leave_one_out mode subtracts the group mean, no division by std."""
    vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = _loo_norm(vals, mode="leave_one_out")
    assert np.allclose(out, [-1.0, 0.0, 1.0])  # mean=2


def test_loo_norm_std_mode_divides():
    """std mode divides by population std."""
    vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = _loo_norm(vals, mode="std")
    expected = (vals - 2.0) / (np.std(vals) + 1e-8)
    assert np.allclose(out, expected)


def test_loo_norm_singleton_is_zero():
    """A single-element group has zero advantage in both modes (no peers)."""
    assert np.allclose(_loo_norm(np.array([5.0], dtype=np.float32),
                                 mode="leave_one_out"), [0.0])
    assert np.allclose(_loo_norm(np.array([5.0], dtype=np.float32),
                                 mode="std"), [0.0])


def test_to_hashable_strings_and_lists():
    assert _to_hashable("abc") == "abc"
    assert _to_hashable(["a", "b"]) == ("a", "b")


def test_are_similar_threshold():
    assert _are_similar("Output: ok", "Output: ok", 0.9) is True
    assert _are_similar("Output: ok", "totally different text here", 0.9) is False


def test_discounted_step_returns_terminal_only():
    """R_k = gamma^(N-k) * outcome for terminal-only reward, k=1..N (1-indexed)."""
    out = _discounted_step_returns(num_steps=3, outcome=1.0, gamma=0.95)
    assert np.allclose(out, [0.95 ** 2, 0.95, 1.0])
    assert np.allclose(_discounted_step_returns(3, 0.0, 0.95), [0.0, 0.0, 0.0])
    assert np.allclose(_discounted_step_returns(1, 0.5, 0.95), [0.5])


def test_build_step_groups_exact_match():
    """Steps with identical anchor strings get the same group id."""
    anchors = [["A", "B"], ["A", "C"]]
    groups = _build_step_groups(anchors, enable_similarity=False, threshold=0.9)
    assert groups[0][0] == groups[1][0]      # both "A"
    assert groups[0][1] != groups[0][0]      # "B" different from "A"
    assert groups[0][1] != groups[1][1]      # "B" != "C"


def test_build_step_groups_similarity():
    """Near-identical anchors cluster together; dissimilar ones stay separate."""
    anchors = [["Output: ok aaaa"], ["Output: ok aaab"], ["completely different xyz"]]
    groups = _build_step_groups(anchors, enable_similarity=True, threshold=0.9)
    assert groups[0][0] == groups[1][0]   # near-identical -> same cluster
    assert groups[2][0] != groups[0][0]   # dissimilar -> different cluster
