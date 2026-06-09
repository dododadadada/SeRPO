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
