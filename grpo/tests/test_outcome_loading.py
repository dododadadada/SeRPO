"""Tests for the new outcome_type parameter in _load_outcomes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from grpo.preprocess.build_dataset import _load_outcomes


def _make_seed_eval(base: Path, seed: int, recs: dict[str, dict]) -> None:
    """Write a fake seed_N/evaluations/train.json. ``recs`` maps task_id to
    a record dict containing success/passes/failures/num_tests fields."""
    seed_dir = base / f"seed_{seed}" / "evaluations"
    seed_dir.mkdir(parents=True)
    payload = {
        "aggregate": {"task_goal_completion": 0.0, "scenario_goal_completion": 0.0},
        "individual": recs,
    }
    (seed_dir / "train.json").write_text(json.dumps(payload))


def test_binary_outcome_uses_success_bool(tmp_path: Path) -> None:
    """binary outcome_type returns 1.0 if success else 0.0."""
    _make_seed_eval(
        tmp_path,
        1,
        {
            "t_pass": {"success": True, "passes": [{}], "failures": [], "num_tests": 1},
            "t_fail_partial": {
                "success": False,
                "passes": [{}, {}, {}],
                "failures": [{}, {}],
                "num_tests": 5,
            },
            "t_fail_total": {
                "success": False, "passes": [], "failures": [{}], "num_tests": 1,
            },
        },
    )
    out = _load_outcomes(tmp_path, 1, "binary")
    assert out == {"t_pass": 1.0, "t_fail_partial": 0.0, "t_fail_total": 0.0}


def test_continuous_outcome_uses_pass_rate(tmp_path: Path) -> None:
    """continuous outcome_type returns passes / num_tests."""
    _make_seed_eval(
        tmp_path,
        1,
        {
            "t_pass": {"success": True, "passes": [{}], "failures": [], "num_tests": 1},
            "t_fail_3_5": {
                "success": False,
                "passes": [{}, {}, {}],
                "failures": [{}, {}],
                "num_tests": 5,
            },
            "t_fail_0_2": {
                "success": False, "passes": [], "failures": [{}, {}], "num_tests": 2,
            },
        },
    )
    out = _load_outcomes(tmp_path, 1, "continuous")
    assert out["t_pass"] == 1.0
    assert out["t_fail_3_5"] == pytest.approx(3 / 5)
    assert out["t_fail_0_2"] == 0.0


def test_continuous_zero_num_tests_yields_zero(tmp_path: Path) -> None:
    """num_tests=0 → 0.0 (avoid div-by-zero)."""
    _make_seed_eval(
        tmp_path,
        1,
        {"t": {"success": False, "passes": [], "failures": [], "num_tests": 0}},
    )
    out = _load_outcomes(tmp_path, 1, "continuous")
    assert out == {"t": 0.0}


def test_unknown_outcome_type_raises(tmp_path: Path) -> None:
    _make_seed_eval(tmp_path, 1, {"t": {"success": True, "num_tests": 1, "passes": [{}], "failures": []}})
    with pytest.raises(ValueError, match="unknown outcome_type"):
        _load_outcomes(tmp_path, 1, "ternary")


def test_real_data_continuous_has_signal_in_failonly() -> None:
    """End-to-end check: in the real Round-0 fail-only set, continuous outcomes
    show cross-seed variance for the majority of tasks."""
    base = Path(
        "/data/minjeong/Autonomous_agent/appworld/experiments/outputs/rollout/round0"
    )
    if not base.exists():
        pytest.skip("real rollout not present")
    from grpo.preprocess.failset import build_failset

    failset = set(build_failset(base))
    rates_per_task: dict[str, list[float]] = {tid: [] for tid in failset}
    for seed_n in range(1, 9):
        outs = _load_outcomes(base, seed_n, "continuous")
        for tid in failset:
            rates_per_task[tid].append(outs.get(tid, 0.0))
    import statistics
    has_variance = sum(
        1 for tid, rates in rates_per_task.items()
        if len(rates) >= 2 and statistics.stdev(rates) > 1e-9
    )
    # We measured empirically: 59/81 tasks have nonzero cross-seed pass_rate variance
    assert has_variance >= 50, (
        f"expected ≥ 50 fail-only tasks with continuous-outcome variance, got {has_variance}"
    )
