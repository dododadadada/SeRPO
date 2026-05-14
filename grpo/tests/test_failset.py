"""Tests for grpo.preprocess.failset."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from grpo.preprocess.failset import build_failset


def _make_seed_eval(base: Path, seed: int, task_outcomes: dict[str, bool]) -> None:
    """Write a fake seed_N/evaluations/train.json mimicking AppWorld format."""
    seed_dir = base / f"seed_{seed}" / "evaluations"
    seed_dir.mkdir(parents=True)
    payload = {
        "aggregate": {"task_goal_completion": 0.0, "scenario_goal_completion": 0.0},
        "individual": {
            tid: {
                "success": ok,
                "difficulty": 1,
                "num_tests": 2,
                "passes": [],
                "failures": [],
            }
            for tid, ok in task_outcomes.items()
        },
    }
    (seed_dir / "train.json").write_text(json.dumps(payload))


def test_failset_strict_zero_of_eight(tmp_path: Path) -> None:
    """Returns only task_ids that fail in ALL 8 seeds."""
    base = tmp_path / "rollout" / "round0"
    base.mkdir(parents=True)
    # task_A: 0/8 success → in failset
    # task_B: 1/8 success (seed_3 succeeds) → not in failset
    # task_C: 0/8 success → in failset
    for s in range(1, 9):
        _make_seed_eval(
            base,
            s,
            {
                "task_A": False,
                "task_B": (s == 3),
                "task_C": False,
            },
        )
    failset = build_failset(base)
    assert sorted(failset) == ["task_A", "task_C"]


def test_failset_missing_seed_raises(tmp_path: Path) -> None:
    """If fewer than 8 seed dirs are present, raise."""
    base = tmp_path / "rollout" / "round0"
    base.mkdir(parents=True)
    for s in (1, 2, 3):  # only 3 seeds
        _make_seed_eval(base, s, {"task_A": False})
    with pytest.raises(ValueError, match="expected 8 seeds"):
        build_failset(base)


def test_failset_real_data() -> None:
    """End-to-end test against real Round-0 data: expect exactly 81 task_ids."""
    base = Path(
        "/data/minjeong/Autonomous_agent/appworld/experiments/outputs/rollout/round0"
    )
    if not base.exists():
        pytest.skip("Round-0 rollout not present")
    failset = build_failset(base)
    assert len(failset) == 81
