from __future__ import annotations
from dataclasses import dataclass, field, asdict
import yaml

@dataclass
class OnlineConfig:
    model_name: str = "Qwen/Qwen3.5-9B"
    M: int = 6
    G: int = 8
    K: int = 1
    N_rounds: int = 76
    dataset: str = "train"
    temperature: float = 1.0
    top_p: float = 1.0
    lr: float = 1e-6
    kl_beta: float = 0.04
    clip_eps: float = 0.2
    grad_clip: float = 1.0
    micro_batch_size: int = 1
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    kl_per_token_cap: float = 10.0
    outlier_logratio_threshold: float = 3.0
    max_masked_fraction: float = 0.05
    kl_halt_threshold: float = 5.0
    ratio_halt_threshold: float = 30.0
    seed: int = 42
    gradient_checkpointing: bool = True
    trainer_gpus: str = "0,1"
    gen_gpus: list = field(default_factory=lambda: ["2"])
    gen_ports: list = field(default_factory=lambda: [8101])
    adapter_dir: str = "grpo_online/runs/v1/adapter_current"
    output_dir: str = "grpo_online/runs/v1"
    rubric_api_model: str = "gpt-4o-mini"
    rubric_api_base: str = ""
    rubric_api_key_env: str = "OPENAI_API_KEY"
    rubric_variant: str = "KS_baseline"

def load_config(path: str) -> OnlineConfig:
    raw = yaml.safe_load(open(path).read()) or {}
    base = asdict(OnlineConfig())
    base.update(raw)
    if isinstance(base.get("lora_target_modules"), list):
        base["lora_target_modules"] = tuple(base["lora_target_modules"])
    return OnlineConfig(**base)
