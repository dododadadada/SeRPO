"""Tests for gigpo.advantage."""
from __future__ import annotations

import numpy as np

from grpo.preprocess.make_advantages import RolloutData
from gigpo.advantage import (
    _are_similar,
    _build_step_groups,
    _discounted_step_returns,
    _loo_norm,
    _to_hashable,
    compute_gigpo_advantage,
)


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


def _mock_gigpo_rollout(outcome, anchors):
    """2-step rollout: step1 tokens [2:5], step2 tokens [6:8]."""
    return RolloutData(
        task_id="t", seed=1,
        input_ids=list(range(10)), attention_mask=[1] * 10,
        response_mask=[0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        step_token_ranges=[(2, 5), (6, 8)],
        segments=[], outcome=float(outcome),
        step_anchor_obs=list(anchors),
    )


def test_gigpo_all_fail_group_is_zero():
    """All outcomes 0 -> A^E=0 and all step returns 0 -> A^S=0 -> all zero."""
    rollouts = [_mock_gigpo_rollout(0, ["A", "B"]) for _ in range(8)]
    compute_gigpo_advantage(rollouts, gamma=0.95, omega=1.0,
                            norm_mode="leave_one_out")
    for r in rollouts:
        assert np.allclose(r.token_adv, 0.0)


def test_gigpo_zero_outside_assistant():
    """response_mask=0 positions stay zero."""
    # NOTE: this relies on the tokenizer contract that step_token_ranges are
    # strict subsets of assistant spans (response_mask==1), so the per-step
    # A^S write never lands on a response_mask==0 token.
    rollouts = [_mock_gigpo_rollout(1 if i < 2 else 0, ["A", "B"])
                for i in range(8)]
    compute_gigpo_advantage(rollouts, gamma=0.95, omega=1.0,
                            norm_mode="leave_one_out")
    for r in rollouts:
        for i, m in enumerate(r.response_mask):
            if m == 0:
                assert r.token_adv[i] == 0.0


def test_gigpo_additive_episode_plus_step():
    """token_adv on a step = A^E + omega*A^S; verify against hand computation."""
    rollouts = [_mock_gigpo_rollout(1 if i < 2 else 0, ["A", "B"])
                for i in range(8)]
    compute_gigpo_advantage(rollouts, gamma=0.95, omega=1.0,
                            norm_mode="leave_one_out")
    outcomes = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    ae = outcomes - outcomes.mean()  # leave_one_out
    g = 0.95
    step1_returns = np.array([g * o for o in outcomes], dtype=np.float32)
    step2_returns = np.array([1.0 * o for o in outcomes], dtype=np.float32)
    as1 = step1_returns - step1_returns.mean()
    as2 = step2_returns - step2_returns.mean()
    for idx, r in enumerate(rollouts):
        assert np.isclose(r.token_adv[2], ae[idx] + 1.0 * as1[idx], atol=1e-5)
        assert np.isclose(r.token_adv[6], ae[idx] + 1.0 * as2[idx], atol=1e-5)


def test_gigpo_singleton_step_group_has_zero_step_adv():
    """A step whose anchor is unique (group size 1) gets A^S=0, so token_adv == A^E."""
    rollouts = [_mock_gigpo_rollout(1 if i < 2 else 0,
                                    ["A", "UNIQUE" if i == 0 else "B"])
                for i in range(8)]
    compute_gigpo_advantage(rollouts, gamma=0.95, omega=1.0,
                            norm_mode="leave_one_out")
    outcomes = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    ae0 = outcomes[0] - outcomes.mean()
    assert np.isclose(rollouts[0].token_adv[6], ae0, atol=1e-5)
