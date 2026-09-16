"""Offline GRPO trainer.

Loads a parquet dataset of trajectories with precomputed per-token advantages,
runs K optimizer steps of clipped GRPO with k3 KL-to-ref via peft.disable_adapter.

Round-1 specifics: round-start LoRA is freshly initialized (zero delta), so
old_logp == ref_logp at step 0; ratio = 1 → loss = pg_loss + β * kl. After step
0 the LoRA delta begins to drift, ratio diverges from 1, and clipping kicks in.

This trainer is single-process (no distributed). For 7B + LoRA + 32k context,
a single 96GB GPU is enough; gradient checkpointing is enabled by default.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo.trainer.dataset import ParquetGRPODataset, collate_pad_right

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    parquet_path: str = ""
    output_dir: str = ""
    K_steps: int = 20
    mini_batch_size: int = 8
    micro_batch_size: int = 1
    # Pre-cache forward batch (old + ref logp). Larger value = faster pre-cache
    # but higher peak GPU memory. Independent of training micro_batch_size.
    precache_micro_batch_size: int = 1
    lr: float = 3e-6                # community-recommended for 9B GRPO
    clip_eps: float = 0.2
    kl_beta: float = 0.04           # TRL default for GRPO
    grad_clip: float = 0.1          # Unsloth recommendation; was 1.0
    lora_rank: int = 32
    lora_alpha: int = 32            # Unsloth: alpha == r
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = field(
        default_factory=lambda: (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        )
    )
    # Optimizer + scheduler. Community 9B-GRPO recipe uses paged AdamW 8bit with
    # cosine schedule + 10% warmup, weight_decay 0.1, beta2 0.99 (not the torch
    # default 0.999) — the lower beta2 makes Adam's second-moment estimate adapt
    # faster, which is more stable for short-horizon RL fine-tuning.
    use_8bit_optim: bool = True
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"  # "cosine" or "constant"
    # QLoRA: load base in 4bit (NF4 + double quantization). Saves VRAM and adds
    # quantization noise that acts as mild regularization. Optional — defaults
    # to False because our 96GB GPUs don't need the VRAM savings.
    load_in_4bit: bool = False
    seed: int = 42
    log_every: int = 1
    gradient_checkpointing: bool = True
    save_every_n_steps: int = 0  # 0 disables intermediate adapter saves
    # Resume from a previously saved LoRA adapter directory (e.g. ".../step_10").
    # Empty string = fresh init. The step number is parsed from the dir name;
    # training continues at that step until K_steps.
    resume_from: str = ""
    # Reuse the on-disk base-policy precache when resuming. This trainer computes
    # old_logp/ref_logp ONCE (= base, since fresh LoRA delta=0) and never refreshes
    # them, so the ORIGINAL run used old_logp=base for every step. Reusing the base
    # cache on resume therefore faithfully continues that same run (and skips the
    # multi-hour re-precache). Default False keeps the conservative re-anchor path.
    reuse_precache_on_resume: bool = False
    # Divergence guard: halt training if loss/kl is non-finite (NaN/Inf) or
    # if kl_loss exceeds this threshold for a single step. 0 disables the KL
    # threshold check (NaN/Inf guard is always on). Typical safe value: 5.0,
    # which catches policy explosions while allowing normal large-batch noise.
    kl_halt_threshold: float = 0.0
    # Fresh LoRA invariant: before the first optimizer step, the policy should
    # match the reference policy. A larger value almost always means stale
    # pre-cache, mismatched batches, or a changed forward path.
    step0_kl_tolerance: float = 1e-3
    # Halt on pathological PPO ratio tails before the update is applied.
    ratio_halt_threshold: float = 20.0
    # Per-token KL cap. On long trajectories (20k+ tokens) Qwen3.5's bf16
    # torch-fallback linear-attention recurrence accumulates error, so a few
    # tokens can show |ref_logp - new_logp| ≈ 20 even at near-fresh init — a
    # numerical artifact, not real policy drift (15/16 sequences stay at
    # kl≈5e-4; one long sequence spikes a single token to k3≈5e6, dragging the
    # mini-batch mean to ~80 and tripping the halt). k3 = exp(d)-d-1 explodes
    # for large d, and .clamp(max=cap) bounds BOTH the metric and the gradient
    # (clamp zeroes grad above the cap → artifact tokens contribute nothing).
    # Normal tokens peak at k3≈0.36, so cap=10 never touches real signal.
    # 0 disables.
    kl_per_token_cap: float = 10.0
    # Outlier-token mask threshold (nats). A response token whose recomputed
    # |new_logp - old_logp| OR |ref_logp - new_logp| exceeds this is treated as
    # a bf16 long-sequence numerical artifact and dropped from ALL loss terms
    # AND all guard metrics (so it can't trip kl_halt / ratio_halt on noise).
    # Measured (diag_logratio_dist) at step 1: mean |log_ratio|=0.03, 99.5% of
    # tokens < 1 nat, but a continuous tail (0.2% > 3 nats, out to ~21) from the
    # long-sequence recurrence. 3.0 masks that 0.2% tail while keeping all real
    # signal. MUST satisfy threshold <= log(ratio_halt_threshold) so the guard
    # never trips on a token the mask let through. 0 disables.
    outlier_logratio_threshold: float = 3.0
    # Real-divergence guard: if more than this fraction of response tokens get
    # masked as artifacts in a step, the policy has genuinely blown up (not just
    # a long-sequence numerical tail — that's ~0.2%). Halts before the update.
    # 0 disables.
    max_masked_fraction: float = 0.05
    # Weights & Biases logging. Empty wandb_project disables.
    # wandb_run_name defaults to the output_dir's basename when empty.
    wandb_project: str = ""
    wandb_run_name: str = ""


def logprobs_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, chunk_size: int = 1024,
) -> torch.Tensor:
    """logits: (B, T, V); labels: (B, T) -> per-token logprob (B, T).

    Chunked along the sequence dim so we never materialize the full (B, T, V)
    float32 softmax tensor — Qwen vocab is ~248k and T can be ~30k, which is
    ~30 GB at fp32 and trivially OOMs an L40S.
    """
    B, T, V = logits.shape
    out = torch.empty((B, T), device=logits.device, dtype=torch.float32)
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        chunk = logits[:, s:e].float()
        # fused log_softmax + gather via cross_entropy(reduction='none')
        nll = F.cross_entropy(
            chunk.reshape(-1, V),
            labels[:, s:e].reshape(-1),
            reduction="none",
        )
        out[:, s:e] = (-nll).view(B, e - s)
    return out


def shift_for_causal_lm(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
):
    """Shift labels/masks/advantages for next-token prediction.

    After the shift, position i in the *_s tensors corresponds to the model
    predicting input_ids[:, i+1] given context input_ids[:, :i+1].
    response_mask[:, i+1] tells whether that predicted token is an assistant
    token (i.e., should contribute to loss). advantages[:, i+1] is the per-
    token advantage for that prediction.
    """
    return (
        input_ids[:, :-1],
        input_ids[:, 1:],         # labels
        attention_mask[:, :-1],
        response_mask[:, 1:],
        advantages[:, 1:],
    )


def _forward_logprobs(model, inputs: torch.Tensor, attn: torch.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=inputs.to(model.device),
                attention_mask=attn.to(model.device))
    return logprobs_from_logits(out.logits, labels.to(model.device))


def _precache_cache_path(cfg: "GRPOConfig") -> Path:
    """Disk path for the pre-cache of (old_logp, ref_logp) tensors.

    Keyed by (base_model, parquet, seed, K_steps, mini_batch_size). At fresh
    LoRA init, old_logp == ref_logp == base model logp on the same trajectory,
    so this cache is the same regardless of LoRA rank/alpha/lr/kl_beta.
    Sibling to the parquet so multiple experiments on the same dataset share.
    """
    parquet = Path(cfg.parquet_path)
    safe_model = cfg.model_name.replace("/", "--")
    # `q4` suffix when 4bit base is enabled — the quantized forward differs
    # numerically from bf16, so the caches are NOT interchangeable.
    quant_tag = "_q4" if cfg.load_in_4bit else ""
    return parquet.parent / (
        f"precache_{safe_model}_{parquet.stem}"
        f"_seed{cfg.seed}_K{cfg.K_steps}_bsz{cfg.mini_batch_size}{quant_tag}.pt"
    )


def _cache_meta(cfg: "GRPOConfig") -> dict:
    parquet = Path(cfg.parquet_path)
    stat = parquet.stat()
    return {
        # v3: added precache_micro_batch_size. Qwen3.5's linear-attention torch
        # fallback forward is batch-size dependent (a sequence's per-token logp
        # changes by up to ~0.36 nats when forwarded in a group of 4 vs alone),
        # so a cache built at precache_micro_batch_size != training
        # micro_batch_size makes old_logp/ref_logp mismatch step-0 new_logp →
        # spurious PPO ratio → divergence. Keying on it forces a rebuild when
        # the grouping changes, instead of silently loading a mismatched cache.
        "version": 3,
        "model_name": cfg.model_name,
        "parquet_path": str(parquet),
        "parquet_size": stat.st_size,
        "parquet_mtime_ns": stat.st_mtime_ns,
        "K_steps": cfg.K_steps,
        "mini_batch_size": cfg.mini_batch_size,
        "precache_micro_batch_size": cfg.precache_micro_batch_size,
        "seed": cfg.seed,
        "load_in_4bit": cfg.load_in_4bit,
        # NOTE: lora_rank / lora_alpha / lora_target_modules are deliberately
        # NOT part of the cache key. At fresh LoRA init the adapter delta is 0
        # (B=0), so old_logp (adapter ON) == ref_logp (adapter OFF) == base-model
        # logp regardless of LoRA shape/coverage (verified: diag_forward_repro
        # "adapter on vs off" = 0.0). Keying on them would force a multi-hour
        # recache for every target_modules/rank sweep even though the cached
        # tensors are identical. The just-saved v3 cache (which still carries
        # these keys) stays valid — _load_valid_precache only checks keys present
        # in the expected meta, so dropping them here is backward-compatible.
    }


def _batches_match_cache(
    cached: list[dict[str, torch.Tensor]],
    batches: list[dict[str, torch.Tensor]],
) -> tuple[bool, str]:
    if len(cached) != len(batches):
        return False, f"batch_count cached={len(cached)} current={len(batches)}"
    for i, (c, b) in enumerate(zip(cached, batches)):
        if c.get("task_ids") != b.get("task_ids"):
            return False, f"task_ids mismatch at batch {i}"
        if c.get("seeds") != b.get("seeds"):
            return False, f"seeds mismatch at batch {i}"
        for key in ("input_ids", "attention_mask", "response_mask", "advantages"):
            if key not in c:
                return False, f"missing {key} in cached batch {i}"
            if not torch.equal(c[key], b[key]):
                return False, f"{key} mismatch at batch {i}"
        T = b["input_ids"].shape[1] - 1
        B = b["input_ids"].shape[0]
        for key in ("old_logp", "ref_logp"):
            if key not in c:
                return False, f"missing {key} in cached batch {i}"
            if tuple(c[key].shape) != (B, T):
                return False, f"{key} shape mismatch at batch {i}"
            if not torch.isfinite(c[key]).all():
                return False, f"non-finite {key} in cached batch {i}"
    return True, "ok"


def _load_valid_precache(
    cache_file: Path,
    cfg: "GRPOConfig",
    batches_to_cache: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]] | None:
    if not cache_file.exists():
        return None
    obj = torch.load(cache_file, map_location="cpu", weights_only=False)
    expected_meta = _cache_meta(cfg)
    if isinstance(obj, dict) and "meta" in obj and "batches" in obj:
        for key, expected in expected_meta.items():
            if obj["meta"].get(key) != expected:
                logger.warning(
                    "ignoring stale pre-cache %s: meta[%s]=%r expected %r",
                    cache_file, key, obj["meta"].get(key), expected,
                )
                return None
        cached = obj["batches"]
    elif isinstance(obj, list):
        logger.warning(
            "ignoring legacy pre-cache %s; cache format predates strict "
            "forward/batch validation",
            cache_file,
        )
        return None
    else:
        logger.warning("ignoring malformed pre-cache %s", cache_file)
        return None

    ok, reason = _batches_match_cache(cached, batches_to_cache)
    if not ok:
        logger.warning("ignoring stale pre-cache %s: %s", cache_file, reason)
        return None
    return cached


def cache_old_and_ref_logprobs(model, batches: list[dict[str, torch.Tensor]],
                               micro_bsz: int) -> list[dict[str, torch.Tensor]]:
    """Precompute old_logp (LoRA enabled) and ref_logp (LoRA disabled) per batch.

    NOTE: caller must set model.train() / disable dropout BEFORE invoking this
    so the cached forward matches the training-step forward exactly.
    (Option-A fix: previously this called model.eval(), causing step-0 kl > 0
    because cache forward path differed from train-mode training forward.)
    """
    cached: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for batch in batches:
            inputs, labels, attn, _, _ = shift_for_causal_lm(
                batch["input_ids"],
                batch["attention_mask"],
                batch["response_mask"],
                batch["advantages"],
            )
            B = inputs.size(0)
            old_lp_chunks = []
            ref_lp_chunks = []
            for s in range(0, B, micro_bsz):
                sl = slice(s, s + micro_bsz)
                old_lp_chunks.append(
                    _forward_logprobs(model, inputs[sl], attn[sl], labels[sl]).cpu()
                )
                with model.disable_adapter():
                    ref_lp_chunks.append(
                        _forward_logprobs(model, inputs[sl], attn[sl], labels[sl]).cpu()
                    )
            old_lp = torch.cat(old_lp_chunks, dim=0)
            ref_lp = torch.cat(ref_lp_chunks, dim=0)
            cached.append({**batch, "old_logp": old_lp, "ref_logp": ref_lp})
    return cached


def grpo_loss_step(model, batch: dict[str, torch.Tensor],
                   cfg: GRPOConfig) -> dict[str, float]:
    """One optimizer step over a mini-batch using gradient accumulation."""
    inputs, labels, attn_s, resp_s, adv_s = shift_for_causal_lm(
        batch["input_ids"], batch["attention_mask"],
        batch["response_mask"], batch["advantages"],
    )
    old_lp_full = batch["old_logp"]
    ref_lp_full = batch["ref_logp"]

    B = inputs.size(0)
    total_loss_w = 0.0
    total_pg_w = 0.0
    total_kl_w = 0.0
    total_resp = 0.0
    total_clip_w = 0.0
    max_ratio = 0.0
    max_abs_log_ratio = 0.0
    total_approx_kl_w = 0.0
    total_masked = 0
    n_micro_batches = (B + cfg.micro_batch_size - 1) // cfg.micro_batch_size

    for s in range(0, B, cfg.micro_batch_size):
        sl = slice(s, s + cfg.micro_batch_size)
        new_lp = _forward_logprobs(model, inputs[sl], attn_s[sl], labels[sl])
        old_lp = old_lp_full[sl].to(model.device)
        ref_lp = ref_lp_full[sl].to(model.device)
        resp = resp_s[sl].to(model.device).float()
        # Clamp advantage to defang outlier per-token rewards.
        adv = adv_s[sl].to(model.device).clamp(-5.0, 5.0)

        raw_log_ratio = new_lp - old_lp
        # Outlier-token mask: drop bf16 long-sequence numerical artifacts (a few
        # tokens on 20k+ token trajectories whose recomputed logp diverges ~20
        # nats from the cached base — not policy drift) from every loss term AND
        # every guard metric, so they can't pollute the gradient or halt on
        # noise. `valid` is the response mask minus artifacts.
        if cfg.outlier_logratio_threshold > 0:
            artifact = (
                (raw_log_ratio.detach().abs() > cfg.outlier_logratio_threshold)
                | ((ref_lp - new_lp).detach().abs() > cfg.outlier_logratio_threshold)
            )
            valid = resp * (~artifact).float()
        else:
            valid = resp
        n_masked = int((resp.bool() & ~valid.bool()).sum().item())

        # PPO clipped surrogate. Clamps prevent single-token exp() blowup:
        # log_ratio in [-20, 20] keeps exp finite; outer ratio.clamp(max=10)
        # is the "dual clip" that bounds the A<0, ratio>>1 case where the
        # standard PPO clip (lower-bound on positive ratio) is one-sided and
        # lets pg_loss go to -inf → loss to +inf.
        log_ratio = raw_log_ratio.clamp(-20.0, 20.0)
        ratio = log_ratio.exp().clamp(max=10.0)
        ratio_clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
        pg_unclipped = ratio * adv
        pg_clipped = ratio_clipped * adv
        pg = -torch.min(pg_unclipped, pg_clipped)
        pg_loss = (pg * valid).sum() / (valid.sum() + 1e-8)

        # k3 KL: exp(r - n) - (r - n) - 1, always ≥ 0. Same exp() blowup risk
        # as ratio above when ref drifts far from new, so clamp identically.
        ref_minus_new = (ref_lp - new_lp).clamp(-20.0, 20.0)
        kl_per_tok = ref_minus_new.exp() - ref_minus_new - 1.0
        # Redundant per-token KL cap (artifacts are already masked by `valid`;
        # this also bounds any legit-but-large token's gradient).
        if cfg.kl_per_token_cap > 0:
            kl_per_tok = kl_per_tok.clamp(max=cfg.kl_per_token_cap)
        kl_loss = (kl_per_tok * valid).sum() / (valid.sum() + 1e-8)

        approx_kl_to_old = (
            raw_log_ratio.clamp(-20.0, 20.0).exp() - 1.0
        ) - raw_log_ratio.clamp(-20.0, 20.0)
        approx_kl_loss = (approx_kl_to_old * valid).sum() / (valid.sum() + 1e-8)
        clipped = ((ratio - 1.0).abs() > cfg.clip_eps).float()
        clipfrac = (clipped * valid).sum() / (valid.sum() + 1e-8)
        active = valid.bool()
        if active.any():
            max_ratio = max(max_ratio, float(ratio[active].detach().float().max().item()))
            max_abs_log_ratio = max(
                max_abs_log_ratio,
                float(raw_log_ratio[active].detach().float().abs().max().item()),
            )

        loss = pg_loss + cfg.kl_beta * kl_loss
        # Scale so the sum across micro-batches approximates the mini-batch mean
        (loss / n_micro_batches).backward()

        # Weight metrics by valid (non-artifact) token count, matching the loss.
        resp_count = max(int(valid.sum().item()), 1)
        total_loss_w += loss.detach().float().item() * resp_count
        total_pg_w += pg_loss.detach().float().item() * resp_count
        total_kl_w += kl_loss.detach().float().item() * resp_count
        total_approx_kl_w += approx_kl_loss.detach().float().item() * resp_count
        total_clip_w += clipfrac.detach().float().item() * resp_count
        total_resp += resp_count
        total_masked += n_masked

    return {
        "loss": total_loss_w / max(total_resp, 1),
        "pg_loss": total_pg_w / max(total_resp, 1),
        "kl_loss": total_kl_w / max(total_resp, 1),
        "approx_kl_to_old": total_approx_kl_w / max(total_resp, 1),
        "clipfrac": total_clip_w / max(total_resp, 1),
        "ratio_max": max_ratio,
        "log_ratio_abs_max": max_abs_log_ratio,
        "resp_tokens": int(total_resp),
        "masked_tokens": int(total_masked),
    }


def run_training(cfg: GRPOConfig) -> None:
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = False
    if cfg.wandb_project:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name or out_dir.name,
                config={k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in asdict(cfg).items()},
                resume="allow",
            )
            use_wandb = True
            logger.info("wandb logging enabled (project=%s)", cfg.wandb_project)
        except Exception as e:
            # ImportError, UsageError (no API key), network failure, etc.
            # Training proceeds without wandb rather than crashing.
            logger.warning("wandb init failed (%s) — continuing without it", e)

    logger.info("loading tokenizer + base model (%s)", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Optional QLoRA: load base in NF4 4-bit. Community recipes (Unsloth) ship
    # this by default for 9B-class models. Adds quantization noise to the frozen
    # base weights that empirically acts as mild regularization for LoRA on
    # hybrid-attention models like Qwen3.5. Disabled by default — turn on via
    # `load_in_4bit: true` in the YAML.
    base_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        base_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        logger.info("QLoRA: loading base in 4bit NF4 + double quant")
    base = AutoModelForCausalLM.from_pretrained(cfg.model_name, **base_kwargs)
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    base.config.use_cache = False  # incompatible with grad ckpt + train

    if cfg.resume_from:
        logger.info("resuming LoRA from %s", cfg.resume_from)
        model = PeftModel.from_pretrained(
            base, cfg.resume_from, is_trainable=True,
        )
        name = Path(cfg.resume_from).name
        if name.startswith("step_"):
            start_step = int(name.split("_")[-1])
        else:
            start_step = 0
        logger.info("resume start_step = %d", start_step)
    else:
        lora_cfg = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=list(cfg.lora_target_modules),
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
        start_step = 0
    model.print_trainable_parameters()

    dataset = ParquetGRPODataset(Path(cfg.parquet_path))
    logger.info("dataset rows: %d", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=cfg.mini_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        collate_fn=lambda b: collate_pad_right(b, tokenizer.pad_token_id),
        drop_last=True,
    )

    # Materialize the dataloader once, then keep only the K batches we will
    # actually consume — caching the full epoch wastes large amounts of GPU
    # time when K << len(loader).
    batches = list(loader)
    if cfg.K_steps <= len(batches):
        batches_to_cache = batches[: cfg.K_steps]
    else:
        # Wrap if K exceeds epoch length (rare; preserves prior loop semantics).
        reps = (cfg.K_steps + len(batches) - 1) // len(batches)
        batches_to_cache = (batches * reps)[: cfg.K_steps]
    # Put model into the SAME state used during training steps BEFORE pre-cache,
    # so cached old_logp/ref_logp match step-0 new_logp exactly (else step 0
    # starts with kl > 0 and PPO ratio noise drives divergence). Requirements:
    #   (a) self.training = True — HF gradient checkpointing guards on it;
    #       without train mode, full activations are stored and 9B OOMs at ~95 GB.
    #   (b) all stochastic noise off — zero any nn.Dropout p so train mode
    #       doesn't perturb the forward output.
    model.train()
    n_dropout = 0
    if hasattr(model, "modules"):
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                if m.p > 0:
                    n_dropout += 1
                m.p = 0.0
    logger.info("disabled %d nn.Dropout layers (set p=0)", n_dropout)

    # Correctness guard: the cached old_logp/ref_logp must be forwarded with the
    # SAME micro-batch grouping used in the training step, because Qwen3.5's
    # linear-attention torch fallback is batch-size dependent. A mismatch makes
    # step-0 kl > 0 (policy != reference at delta=0) and diverges. Warn loudly.
    if cfg.precache_micro_batch_size != cfg.micro_batch_size:
        logger.warning(
            "precache_micro_batch_size=%d != micro_batch_size=%d — cached "
            "old/ref log-probs will NOT match the training forward on "
            "batch-size-dependent models (Qwen3.5). Expect step-0 kl > 0 and "
            "divergence. Set them equal.",
            cfg.precache_micro_batch_size, cfg.micro_batch_size,
        )

    # Pre-cache: try disk cache first (only valid for fresh init, since
    # resume_from has a drifted policy whose old_logp differs from base).
    cache_file = (
        _precache_cache_path(cfg)
        if (not cfg.resume_from or cfg.reuse_precache_on_resume)
        else None
    )
    cached = None
    if cache_file is not None:
        cached = _load_valid_precache(cache_file, cfg, batches_to_cache)
        if cached is not None:
            logger.info("loaded %d validated cached mini-batches from %s",
                        len(cached), cache_file)
    if cached is None:
        logger.info(
            "pre-caching old + ref log-probs for %d mini-batches "
            "(precache_micro_batch_size=%d) ...",
            len(batches_to_cache), cfg.precache_micro_batch_size,
        )
        cached = cache_old_and_ref_logprobs(
            model, batches_to_cache, cfg.precache_micro_batch_size,
        )
        logger.info("cached %d mini-batches", len(cached))
        if cache_file is not None:
            try:
                torch.save({"meta": _cache_meta(cfg), "batches": cached}, cache_file)
                logger.info("saved pre-cache to %s", cache_file)
            except Exception as e:
                logger.warning("failed to save pre-cache (%s) — continuing", e)

    # Optimizer — community 9B recipe: paged AdamW 8bit, beta2=0.99 (faster
    # second-moment adaptation than torch's 0.999 default → safer early RL
    # steps), weight_decay 0.1.
    trainable = [p for p in model.parameters() if p.requires_grad]
    if cfg.use_8bit_optim:
        try:
            from bitsandbytes.optim import PagedAdamW8bit
            optimizer = PagedAdamW8bit(
                trainable, lr=cfg.lr,
                betas=(cfg.adam_beta1, cfg.adam_beta2),
                weight_decay=cfg.weight_decay,
            )
            logger.info(
                "optimizer: PagedAdamW8bit lr=%g betas=(%g,%g) wd=%g",
                cfg.lr, cfg.adam_beta1, cfg.adam_beta2, cfg.weight_decay,
            )
        except ImportError:
            logger.warning(
                "bitsandbytes is not installed; falling back to torch AdamW"
            )
            optimizer = torch.optim.AdamW(
                trainable, lr=cfg.lr,
                betas=(cfg.adam_beta1, cfg.adam_beta2),
                weight_decay=cfg.weight_decay,
            )
    else:
        optimizer = torch.optim.AdamW(
            trainable, lr=cfg.lr,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            weight_decay=cfg.weight_decay,
        )
        logger.info(
            "optimizer: AdamW lr=%g betas=(%g,%g) wd=%g",
            cfg.lr, cfg.adam_beta1, cfg.adam_beta2, cfg.weight_decay,
        )

    # LR scheduler — community 9B recipe: cosine with 10% warmup. Warmup keeps
    # effective per-param lr small while Adam's second-moment estimate fills in,
    # preventing the step-2/3 explosion we observed with constant lr.
    if cfg.lr_scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        num_warmup = int(cfg.warmup_ratio * cfg.K_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=cfg.K_steps,
        )
        logger.info(
            "scheduler: cosine, warmup_steps=%d / total=%d", num_warmup, cfg.K_steps,
        )
    else:
        scheduler = None
        logger.info("scheduler: none (constant lr)")

    # Load existing metrics if resuming so the final metrics.jsonl is contiguous.
    metrics_path = out_dir / "metrics.jsonl"
    if start_step > 0 and metrics_path.exists():
        with metrics_path.open() as f:
            metrics_log: list[dict[str, float | int]] = [
                json.loads(ln) for ln in f if ln.strip()
            ]
        logger.info("loaded %d existing metrics entries", len(metrics_log))
    else:
        metrics_log = []

    step_iter = start_step
    halted = False
    while step_iter < cfg.K_steps and not halted:
        # cached has exactly K_steps mini-batches; on resume we skip those
        # already consumed (cached[:start_step]) and continue from cached[step_iter].
        for batch in cached[step_iter:]:
            if step_iter >= cfg.K_steps:
                break
            optimizer.zero_grad()
            metrics = grpo_loss_step(model, batch, cfg)
            # Divergence guard — runs BEFORE optimizer.step() so the bad update
            # is discarded and the previously saved ckpt remains the last good
            # state.
            if not (
                math.isfinite(metrics["loss"]) and math.isfinite(metrics["kl_loss"])
            ):
                logger.error(
                    "step %d non-finite metrics (loss=%s kl=%s) — halting",
                    step_iter, metrics["loss"], metrics["kl_loss"],
                )
                halted = True
                break
            if (
                cfg.kl_halt_threshold > 0
                and metrics["kl_loss"] > cfg.kl_halt_threshold
            ):
                logger.error(
                    "step %d kl=%.3f exceeded halt threshold %.1f — halting",
                    step_iter, metrics["kl_loss"], cfg.kl_halt_threshold,
                )
                halted = True
                break
            if (
                not cfg.resume_from
                and step_iter == 0
                and cfg.step0_kl_tolerance > 0
                and metrics["kl_loss"] > cfg.step0_kl_tolerance
            ):
                logger.error(
                    "step 0 kl=%.6f exceeded fresh-policy tolerance %.6f — "
                    "pre-cache/forward mismatch; halting before optimizer step",
                    metrics["kl_loss"], cfg.step0_kl_tolerance,
                )
                halted = True
                break
            if (
                cfg.ratio_halt_threshold > 0
                and metrics.get("log_ratio_abs_max", 0.0)
                > math.log(cfg.ratio_halt_threshold)
            ):
                logger.error(
                    "step %d max |log_ratio|=%.3f exceeded log(%g) — halting",
                    step_iter, metrics.get("log_ratio_abs_max", 0.0),
                    cfg.ratio_halt_threshold,
                )
                halted = True
                break
            if cfg.max_masked_fraction > 0:
                _masked = metrics.get("masked_tokens", 0)
                _total = metrics["resp_tokens"] + _masked
                _frac = _masked / max(_total, 1)
                if _frac > cfg.max_masked_fraction:
                    logger.error(
                        "step %d masked_fraction=%.4f (%d/%d) exceeded %.4f — "
                        "likely real divergence (not a numerical tail); halting",
                        step_iter, _frac, _masked, _total, cfg.max_masked_fraction,
                    )
                    halted = True
                    break
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip,
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            metrics["grad_norm"] = float(grad_norm)
            metrics["lr"] = optimizer.param_groups[0]["lr"]
            metrics["step"] = step_iter
            metrics_log.append(metrics)
            if use_wandb:
                wandb.log(
                    {k: v for k, v in metrics.items() if k != "step"},
                    step=step_iter,
                )
            if step_iter % cfg.log_every == 0:
                logger.info(
                    "step %d  loss=%.4f  pg=%.4f  kl=%.4f  old_kl=%.4f  "
                    "clip=%.3f  max|lr|=%.2f  resp_toks=%d  masked=%d",
                    step_iter,
                    metrics["loss"], metrics["pg_loss"], metrics["kl_loss"],
                    metrics.get("approx_kl_to_old", 0.0),
                    metrics.get("clipfrac", 0.0),
                    metrics.get("log_ratio_abs_max", 0.0),
                    metrics["resp_tokens"],
                    metrics.get("masked_tokens", 0),
                )
            step_iter += 1
            if cfg.save_every_n_steps > 0 and step_iter % cfg.save_every_n_steps == 0:
                inter_dir = out_dir / f"step_{step_iter}"
                model.save_pretrained(inter_dir)
                metrics_path.write_text(
                    "\n".join(json.dumps(m) for m in metrics_log) + "\n"
                )
                logger.info("saved intermediate ckpt + metrics → %s", inter_dir)

    model.save_pretrained(out_dir)
    (out_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(m) for m in metrics_log) + "\n"
    )
    cfg_dict = asdict(cfg)
    cfg_dict["lora_target_modules"] = list(cfg_dict["lora_target_modules"])
    (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    logger.info("saved LoRA + metrics + config to %s", out_dir)
    if use_wandb:
        wandb.finish()
