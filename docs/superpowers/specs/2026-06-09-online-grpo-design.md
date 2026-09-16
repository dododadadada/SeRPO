# Online (semi-online) GRPO for Qwen3.5-9B — Design

**Date:** 2026-06-09
**Status:** approved (brainstorming) → pending implementation plan

## Goal

Add an **on-policy / semi-online** GRPO training loop alongside the existing
offline pipeline, so the rollout policy tracks the trainer closely instead of
being refreshed once per round (offline iterative). Reuse the offline trainer,
advantage code, rubric reward, and AppWorld rollout — only the orchestration is
new. Target the same model/task as everything else: Qwen3.5-9B + LoRA on AppWorld,
SeRPO (per-segment rubric, KS_baseline) reward.

## Why semi-online, option "A" (small round)

Chosen for **maximum on-policy-ness** (the point of going online vs our offline
iterative): each round rolls out a SMALL batch with the current policy, takes
**K=1–2** gradient steps, then re-syncs. More distinctly "online" than a large
round (which would resemble the offline iterative loop). Accepted cost: lower
throughput, because peft generation is serial/slow and we extract few updates per
rollout. Off-policy drift is not the concern (lr=1e-6 → ~0.01 KL/step).

## GPU layout — 3 GPUs

| Component | GPU | Notes |
|---|---|---|
| Trainer (9B base + LoRA + optimizer + grad) | 0, 1 | device_map across 2 (peak ~124 GB); holds model in-process for grad |
| Generation: peft server(s) (base + current LoRA) | 2 | OpenAI-compatible endpoint; AppWorld rollout REQUIRES an HTTP endpoint, so generation cannot be in the trainer process |
| Reward judge (rubric segmentation + contribution) | — | **external API** (key-based); no GPU |
| AppWorld environment | CPU | simulated apps |

Trainer and generator are separate processes that communicate via **disk** (the
trainer writes the LoRA adapter; the generator hot-reloads it).

## The loop (one round)

```
for round in 1..N:
  1. ROLLOUT   pick M train tasks; generate G=8 samples each via the peft
               server(s) running the CURRENT LoRA (AppWorld `appworld run`).
               → trajectories (lm_calls.jsonl + train.json outcome).
  2. REWARD    score the NEW trajectories with the rubric scorer (API judge)
               → per-segment contributions (KS_baseline prompt).
  3. ADVANTAGE serpo per-segment z-score over the group, in-memory
               (make_advantages.compute_serpo_advantage) → batch tensors.
  4. TRAIN     recompute old/ref logp on the trainer's peft model; take K=1–2
               grad steps (offline_trainer.grpo_loss_step, with all v3i
               stabilization: outlier mask, kl cap, etc.); save LoRA adapter.
  5. SYNC      POST /reload_adapter to the peft server(s) → hot-swap the new
               adapter (~seconds; no merge/graft/vLLM — peft serves LoRA directly).
```

## Parameters (defaults; configurable)

- `M` tasks/round = **6** (range 4–8)
- `G` samples/task = **8** (locked; matches all prior work)
- `K` grad steps/round = **1** (most on-policy; 2 allowed)
- `N` rounds = **budget-driven** (wall-clock or until an effective-step target)
- Trainer HPs = **v3i** (lr 1e-6, kl_beta 0.04, all-linear LoRA r32, constant lr,
  outlier_logratio_threshold 3, kl_per_token_cap 10, max_masked_fraction 0.05)
- Rubric judge = **external API model** (config-specified; must match the
  KS_baseline rubric prompt). Provided via API key.

## Components — `grpo_online/`

| File | Responsibility | Reuses |
|---|---|---|
| `online_loop.py` | orchestrator: the round loop; owns task sampling, round bookkeeping, budget/stop | — |
| `gen_server.py` | peft server + **`/reload_adapter`** endpoint (hot-swap LoRA in place) | extends `grpo/eval/peft_chat_server.py` |
| `online_trainer.py` | holds model/optimizer across rounds; `train_on_batch(batch, K)` | wraps `grpo/trainer/offline_trainer.py` (`grpo_loss_step`, model/precache setup) |
| `reward.py` | new trajectories → rubric API judge → contributions; build serpo advantages in-memory | `rubric_reward/`, `grpo/preprocess/make_advantages.py`, `tokenize_trajectory.py` |
| `config/online_v1.yaml` | M/G/K/N, trainer HPs, server ports/GPUs, rubric API model | — |

## Data flow / interfaces

- **Trainer ↔ Generator:** trainer writes `…/online/adapter_current/` (LoRA);
  generator's `/reload_adapter` loads it via `PeftModel.load_adapter` + `set_adapter`.
- **Rollout → Reward:** AppWorld writes per-task `lm_calls.jsonl` + `train.json`;
  reward reads those, calls the API judge, returns contributions keyed by (task,seed).
- **Reward → Train:** in-memory list of RolloutData → tokenize → compute_serpo_advantage
  → collated batch (same shape the offline trainer's grpo_loss_step expects:
  input_ids, attention_mask, response_mask, advantages, old_logp, ref_logp).
- old_logp/ref_logp are recomputed in the trainer each round (no disk precache;
  the policy changes every round so a static cache is invalid).

## Risks & mitigations

- **peft generation is serial/slow** → run several peft servers on GPU 2 and shard
  the M×G rollouts across them; each reloads the adapter on sync (cheap).
- **rubric API latency/cost** → batch the judge calls; cost scales with M×G×rounds;
  surface per-round cost in logs.
- **off-policy within K steps** → negligible at lr=1e-6 (~0.01 KL/step); PPO clip +
  outlier mask already handle it (same guards as offline).
- **trainer model held in train mode** for grad-ckpt → old/ref recompute under
  no_grad in the same process is consistent (verified pattern in offline trainer).
- **GPU contention on shared box (8 users)** → pin CUDA_VISIBLE_DEVICES; consider a
  startup memory reservation so precache phase doesn't look free (the OOM lesson).

## Out of scope (YAGNI)

- vLLM generation / weight-update API (blocked by vLLM+Qwen3.5+LoRA bug; peft
  hot-reload replaces it).
- Concurrent gen+train overlap (rounds are sequential in option A).
- Multi-node / full fine-tune (LoRA only).
- A verl/OpenRLHF port (reuse our own trainer instead).

## Success criteria

1. The loop runs N rounds without divergence (guards hold), syncing the policy to
   the generator each round.
2. Produces a checkpoint curve evaluable with the existing `eval_*_ckpt.sh` path.
3. Comparable to offline SeRPO (step_50 = 26.3 dev / 22.0 test) to answer
   "does on-policy online beat offline iterative?".
