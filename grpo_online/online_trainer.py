"""OnlineTrainer: persistent policy + optimizer for online (multi-round) GRPO.

Unlike grpo.trainer.offline_trainer.run_training (a one-shot job that builds the
model, runs K steps, and exits), OnlineTrainer holds the peft model and AdamW
optimizer across rounds. Each round calls train_on_batch(rollouts, K), which:

  1. collates the freshly-generated RolloutData into the padded batch dict that
     grpo_loss_step expects (input_ids/attention_mask/response_mask/advantages),
  2. recomputes old_logp (adapter ON) and ref_logp (adapter OFF) fresh, no_grad
     — the same precache pattern the offline trainer uses, but per-round instead
     of once, because the policy has drifted between rounds,
  3. runs K grad steps of grpo_loss_step (PPO clip + k3 KL + outlier mask + KL
     cap — the proven v3i stabilization).

save_adapter(path) writes the LoRA; lora_snapshot() returns current LoRA tensors.
"""
from __future__ import annotations

import logging
import math

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo.trainer.offline_trainer import (
    GRPOConfig,
    _forward_logprobs,
    grpo_loss_step,
    shift_for_causal_lm,
)
from grpo.trainer.dataset import collate_pad_right

logger = logging.getLogger("online")


class OnlineTrainer:
    def __init__(self, model_name, lora_target_modules, lr=1e-6, lora_rank=32,
                 lora_alpha=32, device_map="auto", gradient_checkpointing=True,
                 resume_adapter=None, **gcfg):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id

        # bf16 on CPU is unsupported for many ops (and pointless for the tiny-
        # model CPU test). Load fp32 on CPU, bf16 everywhere else.
        dtype = torch.float32 if device_map == "cpu" else torch.bfloat16
        base = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map=device_map,
        )
        if gradient_checkpointing:
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()
        base.config.use_cache = False

        if resume_adapter:
            # Continue training from a saved adapter (e.g. ckpt of an earlier
            # round). is_trainable=True keeps the LoRA params requiring grad so
            # the optimizer (built below) updates them. The adapter's own
            # adapter_config.json supplies rank/targets, so they need not match
            # the args. Optimizer (Adam) state restarts fresh — negligible.
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                base, resume_adapter, is_trainable=True)
        else:
            self.model = get_peft_model(base, LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha,
                target_modules=list(lora_target_modules),
                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
        self.model.train()

        self.cfg = GRPOConfig(
            model_name=model_name, lr=lr, lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=tuple(lora_target_modules),
            gradient_checkpointing=gradient_checkpointing, **gcfg)

        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr)

    def _chunked_logprobs(self, inputs, attn, labels, micro):
        # Qwen3.5's torch-fallback linear attention is batch-size dependent: a
        # sequence's per-token logp differs when forwarded in a group vs alone.
        # old_logp/ref_logp MUST be computed with the SAME micro-batch chunking
        # that grpo_loss_step uses for new_lp, else step-0 PPO ratio != 1 and
        # training diverges. Mirror offline_trainer.cache_old_and_ref_logprobs.
        chunks = []
        B = inputs.size(0)
        for s in range(0, B, micro):
            sl = slice(s, s + micro)
            chunks.append(
                _forward_logprobs(self.model, inputs[sl], attn[sl], labels[sl]).cpu()
            )
        return torch.cat(chunks, dim=0)

    def _collate(self, rollouts):
        # collate_pad_right operates on torch tensors (it reads .size(0)/.dtype),
        # but RolloutData carries python lists / a numpy token_adv array, so wrap
        # each field in a tensor first.
        batch = collate_pad_right([{
            "input_ids": torch.tensor(list(r.input_ids), dtype=torch.long),
            "attention_mask": torch.tensor(list(r.attention_mask), dtype=torch.long),
            "response_mask": torch.tensor(list(r.response_mask), dtype=torch.long),
            "advantages": torch.tensor(r.token_adv.tolist(), dtype=torch.float32),
            "task_id": r.task_id,
            "seed": r.seed,
        } for r in rollouts], self.tok.pad_token_id)

        inputs, labels, attn, _, _ = shift_for_causal_lm(
            batch["input_ids"], batch["attention_mask"],
            batch["response_mask"], batch["advantages"])
        micro = self.cfg.micro_batch_size
        with torch.no_grad():
            old = self._chunked_logprobs(inputs, attn, labels, micro)
            with self.model.disable_adapter():
                ref = self._chunked_logprobs(inputs, attn, labels, micro)
        batch["old_logp"] = old
        batch["ref_logp"] = ref
        return batch

    def _should_skip_step(self, metrics) -> str | None:
        """v3i pre-step guards (mirror offline_trainer.run_training:689-725).

        Returns a human-readable reason string if the optimizer step must be
        skipped to avoid corrupting the in-memory LoRA, else None. grpo_loss_step
        returns `loss` as a python float, so isfinite is checked on the float.
        """
        # 1) NaN/Inf loss.
        if not math.isfinite(float(metrics.get("loss", 0.0))):
            return "non-finite loss (NaN/Inf)"
        # 2) Masked-fraction guard. grpo_loss_step does not return the fraction
        #    directly; reconstruct it from masked_tokens / (resp_tokens + masked)
        #    exactly as the offline loop does (offline_trainer.py:714-725).
        cap = getattr(self.cfg, "max_masked_fraction", 0.0)
        if cap > 0:
            masked = metrics.get("masked_tokens", 0)
            total = metrics.get("resp_tokens", 0) + masked
            frac = masked / max(total, 1)
            if frac > cap:
                return (f"masked_fraction={frac:.4f} ({masked}/{total}) "
                        f"exceeded {cap:.4f}")
        # 3) Ratio halt (offline_trainer.py:702-713): max |log_ratio| vs
        #    log(ratio_halt_threshold).
        rht = getattr(self.cfg, "ratio_halt_threshold", 0.0)
        if rht > 0:
            lrm = metrics.get("log_ratio_abs_max", 0.0)
            if lrm > math.log(rht):
                return (f"max |log_ratio|={lrm:.3f} exceeded "
                        f"log({rht:g})={math.log(rht):.3f}")
        return None

    def train_on_batch(self, rollouts, K=1):
        batch = self._collate(rollouts)
        metrics: dict = {}
        for _ in range(K):
            self.opt.zero_grad()
            metrics = grpo_loss_step(self.model, batch, self.cfg)
            skip_reason = self._should_skip_step(metrics)
            if skip_reason is not None:
                # Discard the (possibly poisoned) accumulated grads WITHOUT
                # stepping — a NaN/bad micro-batch must not corrupt the LoRA.
                self.opt.zero_grad()
                logger.warning(
                    "skipping optimizer step: %s "
                    "(loss=%.4g kl=%.4g ratio_max=%.4g log_ratio_abs_max=%.4g "
                    "masked=%d/%d)",
                    skip_reason, float(metrics.get("loss", 0.0)),
                    float(metrics.get("kl_loss", 0.0)),
                    float(metrics.get("ratio_max", 0.0)),
                    float(metrics.get("log_ratio_abs_max", 0.0)),
                    int(metrics.get("masked_tokens", 0)),
                    int(metrics.get("resp_tokens", 0))
                    + int(metrics.get("masked_tokens", 0)),
                )
                metrics["step_skipped"] = True
                # Stop the K-loop: subsequent grad steps would build on the same
                # diverged batch and the offline trainer halts here too.
                break
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.cfg.grad_clip)
            self.opt.step()
            metrics["step_skipped"] = False
        return metrics

    def lora_snapshot(self):
        return {n: p.detach().clone()
                for n, p in self.model.named_parameters() if "lora_" in n}

    def save_adapter(self, path):
        self.model.save_pretrained(path)
