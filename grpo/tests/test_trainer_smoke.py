"""Smoke test for the offline GRPO trainer.

Verifies wiring (data load → forward → loss → backward → save) on a tiny
synthetic parquet using Qwen2.5-0.5B-Instruct. The 7B model is the production
target but takes too long to download + load for a quick smoke check; this
test exercises the same code paths on the same architecture family.

Marked @pytest.mark.gpu — only runs when explicitly selected:
  pytest grpo/tests/test_trainer_smoke.py -v -m gpu
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from grpo.trainer.offline_trainer import GRPOConfig, run_training


@pytest.mark.gpu
def test_trainer_runs_two_steps(tmp_path: Path) -> None:
    """Build a tiny parquet, run K=2 optimizer steps, verify no NaN + checkpoint saved."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(16):
        T = int(32 + rng.integers(0, 16))
        input_ids = rng.integers(0, 1000, size=T).tolist()
        # middle block marked as "response" (loss applies there)
        resp = [0] * 8 + [1] * (T - 16) + [0] * 8
        adv = [(0.1 if m else 0.0) for m in resp]
        rows.append(
            {
                "task_id": f"t{i}",
                "seed": i,
                "input_ids": input_ids,
                "attention_mask": [1] * T,
                "response_mask": resp,
                "advantages": adv,
                "outcome": 0,
                "num_steps": 1,
            }
        )
    parquet_path = tmp_path / "tiny.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)

    cfg = GRPOConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        parquet_path=str(parquet_path),
        output_dir=str(tmp_path / "ckpt"),
        K_steps=2,
        mini_batch_size=8,
        micro_batch_size=1,
        lr=1e-6,
        clip_eps=0.2,
        kl_beta=0.01,
        gradient_checkpointing=False,  # not needed for 0.5B; faster
    )
    run_training(cfg)

    metrics_path = tmp_path / "ckpt" / "metrics.jsonl"
    assert metrics_path.exists()
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    assert len(metrics) == 2
    for m in metrics:
        assert m["loss"] == m["loss"]  # NaN check
        assert m["kl_loss"] >= -1e-3  # k3 KL non-negative

    # LoRA adapter saved
    assert (tmp_path / "ckpt" / "adapter_config.json").exists()
