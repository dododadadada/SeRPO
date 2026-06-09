import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from grpo_online.gen_server import reload_adapter_into

def _tiny_peft(tmp_path):
    m = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    pm = get_peft_model(m, LoraConfig(r=4, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM"))
    return pm

def test_reload_adapter_swaps_weights(tmp_path):
    pm = _tiny_peft(tmp_path)
    a = tmp_path / "A"; pm.save_pretrained(str(a))
    for n, p in pm.named_parameters():
        if "lora_B" in n: p.data.add_(1.0)
    b = tmp_path / "B"; pm.save_pretrained(str(b))
    pm.load_adapter(str(a), adapter_name="default", is_trainable=False)
    pm.set_adapter("default")
    before = [p.detach().clone() for n, p in pm.named_parameters() if "lora_B" in n][0]
    reload_adapter_into(pm, str(b))
    after = [p for n, p in pm.named_parameters() if "lora_B" in n][0]
    assert not torch.allclose(before, after)
