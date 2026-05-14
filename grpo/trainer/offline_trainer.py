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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
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
    lr: float = 1e-6
    clip_eps: float = 0.2
    kl_beta: float = 0.01
    grad_clip: float = 1.0
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = field(
        default_factory=lambda: (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        )
    )
    seed: int = 42
    log_every: int = 1
    gradient_checkpointing: bool = True


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits: (B, T, V); labels: (B, T) -> per-token logprob (B, T)."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


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
        attention_mask[:, 1:],
        response_mask[:, 1:],
        advantages[:, 1:],
    )


def _forward_logprobs(model, inputs: torch.Tensor, attn: torch.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=inputs.to(model.device),
                attention_mask=attn.to(model.device))
    return logprobs_from_logits(out.logits, labels.to(model.device))


def cache_old_and_ref_logprobs(model, batches: list[dict[str, torch.Tensor]],
                               micro_bsz: int) -> list[dict[str, torch.Tensor]]:
    """Precompute old_logp (LoRA enabled) and ref_logp (LoRA disabled) per batch."""
    cached: list[dict[str, torch.Tensor]] = []
    model.eval()
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
    n_micro_batches = (B + cfg.micro_batch_size - 1) // cfg.micro_batch_size

    for s in range(0, B, cfg.micro_batch_size):
        sl = slice(s, s + cfg.micro_batch_size)
        new_lp = _forward_logprobs(model, inputs[sl], attn_s[sl], labels[sl])
        old_lp = old_lp_full[sl].to(model.device)
        ref_lp = ref_lp_full[sl].to(model.device)
        resp = resp_s[sl].to(model.device).float()
        adv = adv_s[sl].to(model.device)

        # PPO clipped surrogate
        log_ratio = new_lp - old_lp
        ratio = log_ratio.exp()
        ratio_clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
        pg_unclipped = ratio * adv
        pg_clipped = ratio_clipped * adv
        pg = -torch.min(pg_unclipped, pg_clipped)
        pg_loss = (pg * resp).sum() / (resp.sum() + 1e-8)

        # k3 KL: exp(r - n) - (r - n) - 1, always ≥ 0
        ref_minus_new = ref_lp - new_lp
        kl_per_tok = ref_minus_new.exp() - ref_minus_new - 1.0
        kl_loss = (kl_per_tok * resp).sum() / (resp.sum() + 1e-8)

        loss = pg_loss + cfg.kl_beta * kl_loss
        # Scale so the sum across micro-batches approximates the mini-batch mean
        (loss / n_micro_batches).backward()

        resp_count = max(int(resp.sum().item()), 1)
        total_loss_w += loss.detach().float().item() * resp_count
        total_pg_w += pg_loss.detach().float().item() * resp_count
        total_kl_w += kl_loss.detach().float().item() * resp_count
        total_resp += resp_count

    return {
        "loss": total_loss_w / max(total_resp, 1),
        "pg_loss": total_pg_w / max(total_resp, 1),
        "kl_loss": total_kl_w / max(total_resp, 1),
        "resp_tokens": int(total_resp),
    }


def run_training(cfg: GRPOConfig) -> None:
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading tokenizer + base model (%s)", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    base.config.use_cache = False  # incompatible with grad ckpt + train

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_target_modules),
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    dataset = ParquetGRPODataset(Path(cfg.parquet_path))
    logger.info("dataset rows: %d", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=cfg.mini_batch_size,
        shuffle=True,
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
    logger.info(
        "pre-caching old + ref log-probs for %d mini-batches "
        "(precache_micro_batch_size=%d) ...",
        len(batches_to_cache), cfg.precache_micro_batch_size,
    )
    cached = cache_old_and_ref_logprobs(
        model, batches_to_cache, cfg.precache_micro_batch_size,
    )
    logger.info("cached %d mini-batches", len(cached))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
    )
    model.train()

    metrics_log: list[dict[str, float | int]] = []
    step_iter = 0
    while step_iter < cfg.K_steps:
        for batch in cached:
            if step_iter >= cfg.K_steps:
                break
            optimizer.zero_grad()
            metrics = grpo_loss_step(model, batch, cfg)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip,
            )
            optimizer.step()
            metrics["step"] = step_iter
            metrics_log.append(metrics)
            if step_iter % cfg.log_every == 0:
                logger.info(
                    "step %d  loss=%.4f  pg=%.4f  kl=%.4f  resp_toks=%d",
                    step_iter,
                    metrics["loss"], metrics["pg_loss"], metrics["kl_loss"],
                    metrics["resp_tokens"],
                )
            step_iter += 1

    model.save_pretrained(out_dir)
    (out_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(m) for m in metrics_log) + "\n"
    )
    cfg_dict = asdict(cfg)
    cfg_dict["lora_target_modules"] = list(cfg_dict["lora_target_modules"])
    (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    logger.info("saved LoRA + metrics + config to %s", out_dir)
