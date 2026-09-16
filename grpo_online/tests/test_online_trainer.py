import torch
from grpo_online.online_trainer import OnlineTrainer
from grpo.preprocess.make_advantages import RolloutData


def _batch():
    r = RolloutData(task_id="t", seed=1, input_ids=[1, 2, 3, 4, 5],
                    attention_mask=[1, 1, 1, 1, 1], response_mask=[0, 1, 1, 1, 0],
                    step_token_ranges=[(1, 4)], segments=[], outcome=1.0)
    import numpy as np
    r.token_adv = np.array([0, 0.5, 0.5, 0.5, 0], dtype="float32")
    return [r]


def _batch_n(n):
    import numpy as np
    out = []
    for i in range(n):
        r = RolloutData(task_id=f"t{i}", seed=i, input_ids=[1, 2, 3, 4, 5],
                        attention_mask=[1, 1, 1, 1, 1],
                        response_mask=[0, 1, 1, 1, 0],
                        step_token_ranges=[(1, 4)], segments=[], outcome=1.0)
        r.token_adv = np.array([0, 0.5, 0.5, 0.5, 0], dtype="float32")
        out.append(r)
    return out


def test_collate_chunks_old_ref_for_batch_gt_one():
    # micro_batch_size default is 1, so a batch of 3 forces the chunking path
    # (3 separate forwards concatenated). old_logp/ref_logp must come back with
    # shape (3, T-1) where T is the padded seq len (5 here -> T-1 = 4).
    tr = OnlineTrainer(model_name="hf-internal-testing/tiny-random-gpt2",
                       lora_target_modules=["c_attn"], device_map="cpu")
    assert tr.cfg.micro_batch_size == 1
    batch = tr._collate(_batch_n(3))
    assert batch["old_logp"].shape == (3, 4)
    assert batch["ref_logp"].shape == (3, 4)


def test_resume_loads_saved_adapter(tmp_path):
    # Train a trainer a step so its LoRA differs from fresh init, save it, then
    # build a new trainer with resume_adapter=that dir. The resumed model's LoRA
    # must equal the saved weights (not a fresh random init), and stay trainable.
    tr = OnlineTrainer(model_name="hf-internal-testing/tiny-random-gpt2",
                       lora_target_modules=["c_attn"], device_map="cpu")
    tr.train_on_batch(_batch(), K=1)
    saved = tr.lora_snapshot()
    adir = tmp_path / "ckpt"
    tr.save_adapter(str(adir))

    tr2 = OnlineTrainer(model_name="hf-internal-testing/tiny-random-gpt2",
                        lora_target_modules=["c_attn"], device_map="cpu",
                        resume_adapter=str(adir))
    resumed = tr2.lora_snapshot()
    assert saved and set(saved) == set(resumed)
    for k in saved:
        assert torch.allclose(saved[k].float(), resumed[k].float(), atol=1e-5)
    assert any(p.requires_grad for n, p in tr2.model.named_parameters() if "lora_" in n)


def test_train_step_updates_and_saves(tmp_path):
    tr = OnlineTrainer(model_name="hf-internal-testing/tiny-random-gpt2",
                       lora_target_modules=["c_attn"], device_map="cpu")
    before = tr.lora_snapshot()
    m = tr.train_on_batch(_batch(), K=1)
    assert "loss" in m and "kl_loss" in m
    after = tr.lora_snapshot()
    assert any(not torch.allclose(before[k], after[k]) for k in before)
    tr.save_adapter(str(tmp_path / "ad"))
    assert (tmp_path / "ad" / "adapter_model.safetensors").exists()


def test_nan_loss_skips_optimizer_step(monkeypatch):
    """A NaN loss from grpo_loss_step must NOT corrupt the in-memory LoRA:
    opt.step() is skipped, weights unchanged, and metrics flag step_skipped."""
    tr = OnlineTrainer(model_name="hf-internal-testing/tiny-random-gpt2",
                       lora_target_modules=["c_attn"], device_map="cpu")
    before = tr.lora_snapshot()

    def fake_step(model, batch, cfg):
        return {"loss": float("nan"), "kl_loss": 0.01, "pg_loss": 0.0,
                "ratio_max": 1.0, "log_ratio_abs_max": 0.0,
                "resp_tokens": 3, "masked_tokens": 0}

    monkeypatch.setattr("grpo_online.online_trainer.grpo_loss_step", fake_step)
    stepped = []
    orig_step = tr.opt.step
    monkeypatch.setattr(tr.opt, "step",
                        lambda *a, **k: (stepped.append(1), orig_step(*a, **k)))

    m = tr.train_on_batch(_batch(), K=1)
    assert m["step_skipped"] is True
    assert stepped == []  # opt.step never called
    after = tr.lora_snapshot()
    assert all(torch.allclose(before[k], after[k]) for k in before)


def test_high_masked_fraction_skips_optimizer_step(monkeypatch):
    tr = OnlineTrainer(model_name="hf-internal-testing/tiny-random-gpt2",
                       lora_target_modules=["c_attn"], device_map="cpu")
    before = tr.lora_snapshot()

    def fake_step(model, batch, cfg):
        # masked_fraction = 90/(10+90) = 0.9 >> default 0.05
        return {"loss": 0.1, "kl_loss": 0.01, "pg_loss": 0.0,
                "ratio_max": 1.0, "log_ratio_abs_max": 0.0,
                "resp_tokens": 10, "masked_tokens": 90}

    monkeypatch.setattr("grpo_online.online_trainer.grpo_loss_step", fake_step)
    m = tr.train_on_batch(_batch(), K=1)
    assert m["step_skipped"] is True
    after = tr.lora_snapshot()
    assert all(torch.allclose(before[k], after[k]) for k in before)


def test_ratio_halt_skips_optimizer_step(monkeypatch):
    tr = OnlineTrainer(model_name="hf-internal-testing/tiny-random-gpt2",
                       lora_target_modules=["c_attn"], device_map="cpu")
    before = tr.lora_snapshot()
    import math

    def fake_step(model, batch, cfg):
        # log_ratio_abs_max above log(ratio_halt_threshold) -> halt
        return {"loss": 0.1, "kl_loss": 0.01, "pg_loss": 0.0,
                "ratio_max": 999.0,
                "log_ratio_abs_max": math.log(tr.cfg.ratio_halt_threshold) + 1.0,
                "resp_tokens": 10, "masked_tokens": 0}

    monkeypatch.setattr("grpo_online.online_trainer.grpo_loss_step", fake_step)
    m = tr.train_on_batch(_batch(), K=1)
    assert m["step_skipped"] is True
    after = tr.lora_snapshot()
    assert all(torch.allclose(before[k], after[k]) for k in before)
