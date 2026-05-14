"""Unit test for the K-only pre-cache slicing in run_training.

We don't run actual training (that needs a model on GPU) — we monkey-patch the
``cache_old_and_ref_logprobs`` function to inspect the slice of batches it
receives. This verifies the slicing logic:
  - K <= num_batches → first K kept
  - K  > num_batches → wrap to length K

CPU-only test.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from grpo.trainer.offline_trainer import GRPOConfig


def _make_parquet(parquet_path: Path, n_rows: int) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_rows):
        T = 16
        input_ids = rng.integers(0, 100, size=T).tolist()
        resp = [0] * 4 + [1] * 8 + [0] * 4
        adv = [(0.1 if m else 0.0) for m in resp]
        rows.append({
            "task_id": f"t{i}", "seed": i,
            "input_ids": input_ids,
            "attention_mask": [1] * T,
            "response_mask": resp,
            "advantages": adv,
            "outcome": 0,
            "num_steps": 1,
        })
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)


def _run_with_cache_intercept(cfg: GRPOConfig) -> int:
    """Run the trainer with model loading + actual training stubbed out.
    Returns the number of mini-batches that pre-cache received."""
    captured: dict[str, int] = {}

    # Stub out cache_old_and_ref_logprobs to just record the batch count.
    def fake_cache(_model, batches, _micro_bsz):
        captured["n_batches"] = len(batches)
        # Return minimal stub batches so the training loop runs
        out = []
        for b in batches:
            T = b["input_ids"].shape[1]
            B = b["input_ids"].shape[0]
            stub = dict(b)
            stub["old_logp"] = torch.zeros((B, T - 1))
            stub["ref_logp"] = torch.zeros((B, T - 1))
            out.append(stub)
        return out

    # Stub out the model + tokenizer loading and the gradient step itself.
    # We only care about pre-cache slicing, not actual learning.
    class _StubModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.config = type("C", (), {"use_cache": False})()
            self._train_params = [torch.nn.Parameter(torch.zeros(1))]
        def eval(self): return self
        def train(self): return self
        def gradient_checkpointing_enable(self): pass
        def enable_input_require_grads(self): pass
        def print_trainable_parameters(self): pass
        def parameters(self): return iter(self._train_params)
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_config.json").write_text("{}")

    from transformers import AutoTokenizer
    real_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    with patch("grpo.trainer.offline_trainer.AutoModelForCausalLM") as MockBase, \
         patch("grpo.trainer.offline_trainer.AutoTokenizer.from_pretrained",
               return_value=real_tok), \
         patch("grpo.trainer.offline_trainer.get_peft_model",
               return_value=_StubModel()), \
         patch("grpo.trainer.offline_trainer.cache_old_and_ref_logprobs",
               side_effect=fake_cache), \
         patch("grpo.trainer.offline_trainer.grpo_loss_step",
               return_value={"loss": 0.0, "pg_loss": 0.0, "kl_loss": 0.0,
                             "resp_tokens": 1}):
        MockBase.from_pretrained.return_value = _StubModel()

        from grpo.trainer.offline_trainer import run_training
        run_training(cfg)
    return captured.get("n_batches", -1)


def test_kbatch_cache_when_k_le_epoch(tmp_path: Path) -> None:
    """K=5 with 10 batches available → cache exactly 5 batches."""
    parquet_path = tmp_path / "data.parquet"
    _make_parquet(parquet_path, n_rows=80)  # 80 rows / 8 batch = 10 batches
    cfg = GRPOConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        parquet_path=str(parquet_path),
        output_dir=str(tmp_path / "ckpt"),
        K_steps=5,
        mini_batch_size=8,
        micro_batch_size=1,
        precache_micro_batch_size=1,
    )
    n_cached = _run_with_cache_intercept(cfg)
    assert n_cached == 5


def test_kbatch_cache_wraps_when_k_gt_epoch(tmp_path: Path) -> None:
    """K=15 with 10 batches available → cache wraps to length 15."""
    parquet_path = tmp_path / "data.parquet"
    _make_parquet(parquet_path, n_rows=80)
    cfg = GRPOConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        parquet_path=str(parquet_path),
        output_dir=str(tmp_path / "ckpt"),
        K_steps=15,
        mini_batch_size=8,
        micro_batch_size=1,
        precache_micro_batch_size=1,
    )
    n_cached = _run_with_cache_intercept(cfg)
    assert n_cached == 15


def test_precache_micro_batch_size_passed_through(tmp_path: Path) -> None:
    """precache_micro_batch_size should be forwarded to cache_old_and_ref_logprobs."""
    captured = {}

    def fake_cache(_model, batches, micro_bsz):
        captured["micro_bsz"] = micro_bsz
        out = []
        for b in batches:
            T = b["input_ids"].shape[1]
            B = b["input_ids"].shape[0]
            stub = dict(b)
            stub["old_logp"] = torch.zeros((B, T - 1))
            stub["ref_logp"] = torch.zeros((B, T - 1))
            out.append(stub)
        return out

    parquet_path = tmp_path / "data.parquet"
    _make_parquet(parquet_path, n_rows=24)
    cfg = GRPOConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        parquet_path=str(parquet_path),
        output_dir=str(tmp_path / "ckpt"),
        K_steps=2,
        mini_batch_size=8,
        micro_batch_size=1,
        precache_micro_batch_size=4,
    )

    class _StubModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.config = type("C", (), {"use_cache": False})()
            self._train_params = [torch.nn.Parameter(torch.zeros(1))]
        def eval(self): return self
        def train(self): return self
        def gradient_checkpointing_enable(self): pass
        def enable_input_require_grads(self): pass
        def print_trainable_parameters(self): pass
        def parameters(self): return iter(self._train_params)
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_config.json").write_text("{}")

    from transformers import AutoTokenizer
    real_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    with patch("grpo.trainer.offline_trainer.AutoModelForCausalLM") as MockBase, \
         patch("grpo.trainer.offline_trainer.AutoTokenizer.from_pretrained",
               return_value=real_tok), \
         patch("grpo.trainer.offline_trainer.get_peft_model",
               return_value=_StubModel()), \
         patch("grpo.trainer.offline_trainer.cache_old_and_ref_logprobs",
               side_effect=fake_cache), \
         patch("grpo.trainer.offline_trainer.grpo_loss_step",
               return_value={"loss": 0.0, "pg_loss": 0.0, "kl_loss": 0.0,
                             "resp_tokens": 1}):
        MockBase.from_pretrained.return_value = _StubModel()
        from grpo.trainer.offline_trainer import run_training
        run_training(cfg)

    assert captured["micro_bsz"] == 4
