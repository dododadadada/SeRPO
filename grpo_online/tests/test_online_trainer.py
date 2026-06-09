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
