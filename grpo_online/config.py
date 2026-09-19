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
    outcome_type: str = "continuous"  # "continuous" (frac tests passed) | "binary"
    # AppWorld worker processes per rollout invocation. Caps how many tasks (and
    # hence concurrent gen-server requests) run at once; raise toward M to keep
    # the concurrent gen server busy. Each worker is a CPU process (~0.8 GB RAM).
    appworld_num_processes: int = 4
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
    # Snapshot the adapter to output_dir/ckpt_round_<n> every ckpt_every rounds
    # (in addition to overwriting adapter_dir each round), so a checkpoint curve
    # survives for later eval. 0 disables snapshotting.
    ckpt_every: int = 5
    rubric_api_model: str = "gpt-4o-mini"
    rubric_api_base: str = ""
    rubric_api_key_env: str = "OPENAI_API_KEY"
    rubric_variant: str = "KS_baseline"
    # Concurrent rubric-judge API calls per round (1 = sequential, as before).
    judge_workers: int = 1
    # Transient-error retries for the judge API (429/5xx/connection): attempts and
    # linear backoff base (sleep = judge_backoff_s * attempt).
    judge_max_attempts: int = 8
    judge_backoff_s: float = 15.0
    # Concurrent `appworld run` invocations, one per seed (1 = sequential, as
    # before). >1 needs per-seed experiment configs; run_rollout generates
    # <experiment>_seed<s>.jsonnet stubs that import the base config. In-flight
    # generation requests = seed_parallelism x min(appworld_num_processes, M).
    seed_parallelism: int = 1
    # Generation backend. "peft": external gen_server.py (transformers.generate,
    # serial), synced via POST /reload_adapter. "vllm": run_online owns a vLLM
    # server (see vllm_gen.py) and hot-swaps the LoRA per round.
    gen_backend: str = "peft"
    # Model name AppWorld requests. Under vllm this is the LoRA's name; the base
    # is served as vllm_base_served_name so requests always hit base+LoRA.
    gen_model_name: str = "Qwen/Qwen3.5-9B"
    vllm_bin: str = ".vllm.venv/bin/vllm"
    vllm_base_served_name: str = "qwen35-9b-base"
    vllm_max_model_len: int = 65536
    vllm_max_num_seqs: int = 64
    vllm_gpu_mem_util: float = 0.85
    vllm_extra_args: list = field(default_factory=list)
    # CUDA toolkit pinned for the vLLM child's JIT linking (see vllm_gen.build_serve_env).
    # Must match the vLLM venv's torch CUDA major version.
    vllm_cuda_home: str = "/usr/local/cuda-13.0"
    # peft tensor-name prefix -> vLLM-side prefix, applied when exporting the
    # adapter for vLLM (see vllm_gen.export_adapter_for_vllm). Default: text-only
    # Qwen3_5ForCausalLM names -> Qwen3_5ForConditionalGeneration names.
    vllm_lora_key_prefix_map: dict = field(default_factory=lambda: {
        "base_model.model.model.": "base_model.model.model.language_model."})

def load_config(path: str) -> OnlineConfig:
    raw = yaml.safe_load(open(path).read()) or {}
    base = asdict(OnlineConfig())
    base.update(raw)
    if isinstance(base.get("lora_target_modules"), list):
        base["lora_target_modules"] = tuple(base["lora_target_modules"])
    return OnlineConfig(**base)
