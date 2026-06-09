"""Integration test for build_dataset orchestrator."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from transformers import AutoTokenizer

from grpo.preprocess.build_dataset import build_dataset_for_task_group

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_LM_CALLS = FIXTURE_DIR / "seed_1_07b42fd_1_lm_calls.jsonl"
FIXTURE_JOINT = FIXTURE_DIR / "joint_07b42fd_1.json"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


def _make_inputs(joint: dict) -> list[dict]:
    """Build 8 fake-rollout inputs by reusing the same lm_calls + joint segments."""
    return [
        {
            "task_id": "07b42fd_1",
            "seed": s,
            "lm_calls_path": FIXTURE_LM_CALLS,
            "joint_record": joint,
            "outcome": 0,
        }
        for s in range(1, 9)
    ]


def test_build_dataset_serpo_writes_parquet(tmp_path, tokenizer):
    joint = json.loads(FIXTURE_JOINT.read_text())
    inputs = _make_inputs(joint)
    out_path = tmp_path / "serpo.parquet"
    n_written = build_dataset_for_task_group(
        inputs, tokenizer=tokenizer, method="serpo", out_path=out_path,
    )
    assert n_written == 8
    df = pq.read_table(out_path).to_pandas()
    assert len(df) == 8
    expected_cols = {
        "task_id", "seed", "input_ids", "attention_mask",
        "response_mask", "advantages", "outcome", "num_steps",
    }
    assert expected_cols.issubset(df.columns)
    row0 = df.iloc[0]
    assert len(row0["input_ids"]) == len(row0["advantages"])
    assert len(row0["input_ids"]) == len(row0["response_mask"])


def test_build_dataset_vanilla_failonly_all_zero_advantage(tmp_path, tokenizer):
    """In fail-only set with outcome=0 for all 8 seeds, vanilla advantage is all zero."""
    joint = json.loads(FIXTURE_JOINT.read_text())
    inputs = _make_inputs(joint)
    out_path = tmp_path / "vanilla.parquet"
    build_dataset_for_task_group(
        inputs, tokenizer=tokenizer, method="vanilla", out_path=out_path,
    )
    df = pq.read_table(out_path).to_pandas()
    for _, row in df.iterrows():
        adv = list(row["advantages"])
        assert all(a == 0.0 for a in adv)


def test_build_dataset_serpo_advantages_have_signal(tmp_path, tokenizer):
    """SeRPO over a group with mixed contributions yields nonzero per-token advantages
    on assistant tokens that fall within scored segments."""
    joint = json.loads(FIXTURE_JOINT.read_text())
    inputs = _make_inputs(joint)
    out_path = tmp_path / "serpo.parquet"
    build_dataset_for_task_group(
        inputs, tokenizer=tokenizer, method="serpo", out_path=out_path,
    )
    df = pq.read_table(out_path).to_pandas()
    # Check at least one non-zero advantage exists in some asst token
    found_signal = False
    for _, row in df.iterrows():
        adv = list(row["advantages"])
        rmask = list(row["response_mask"])
        for a, m in zip(adv, rmask):
            if m == 1 and abs(a) > 1e-6:
                found_signal = True
                break
        if found_signal:
            break
    assert found_signal, "expected at least one nonzero advantage on an assistant token"


def test_gigpo_failonly_is_rejected(tmp_path):
    from grpo.preprocess.build_dataset import run_build
    with pytest.raises(ValueError, match="gigpo|failonly"):
        run_build(
            method="gigpo",
            condition="failonly",
            rollout_dir=tmp_path / "nope",
            joint_dir=tmp_path / "nope",
            output_dir=tmp_path / "out",
            outcome_type="binary",
        )


def test_gigpo_group_produces_nonzero_advantage(tmp_path, tokenizer):
    """GiGPO over a group with differing outcomes (one success, one fail) yields
    at least one nonzero per-token advantage (A^E is nonzero under LOO when the
    group outcomes differ)."""
    joint = json.loads(FIXTURE_JOINT.read_text())
    inputs = [
        {
            "task_id": "07b42fd_1",
            "seed": s,
            "lm_calls_path": FIXTURE_LM_CALLS,
            "joint_record": joint,
            "outcome": 1 if s == 1 else 0,  # one success, rest fail
        }
        for s in range(1, 9)
    ]
    out_path = tmp_path / "gigpo.parquet"
    n_written = build_dataset_for_task_group(
        inputs, tokenizer=tokenizer, method="gigpo", out_path=out_path,
    )
    assert n_written == 8
    df = pq.read_table(out_path).to_pandas()
    found_signal = False
    for _, row in df.iterrows():
        adv = list(row["advantages"])
        rmask = list(row["response_mask"])
        for a, m in zip(adv, rmask):
            if m == 1 and abs(a) > 1e-6:
                found_signal = True
                break
        if found_signal:
            break
    assert found_signal, "expected at least one nonzero advantage on an assistant token"


def test_build_dataset_skips_oversize(tmp_path, tokenizer, monkeypatch):
    """If a tokenized trajectory exceeds MAX_TOKENS, the whole group is dropped."""
    import grpo.preprocess.build_dataset as bd
    monkeypatch.setattr(bd, "MAX_TOKENS", 100)  # lower than fixture (~8.7k)
    joint = json.loads(FIXTURE_JOINT.read_text())
    inputs = _make_inputs(joint)
    out_path = tmp_path / "oversize.parquet"
    n = bd.build_dataset_for_task_group(
        inputs, tokenizer=tokenizer, method="serpo", out_path=out_path,
    )
    assert n == 0
    assert not out_path.exists()  # no parquet written for empty group
