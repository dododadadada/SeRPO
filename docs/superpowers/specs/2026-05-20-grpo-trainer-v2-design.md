# GRPO Trainer v2 — Design

**Date**: 2026-05-20
**Variant**: `vanilla_full_continuous_v2`
**Base**: v1 (`vanilla_full_continuous`, round-1, K=40)

## Why v2

The v1 run (`grpo/ckpts/vanilla_full_continuous/round1`, 40 steps) shows:

1. **Data underutilization** — `batches[:K_steps]` slicing in `offline_trainer.py:231` consumes only 40/89 batches (320/712 trajectories ≈ 45%).
2. **Loss high-variance** — per-step pg_loss sign-flips (+1.75 → −0.50 → +0.87) across 40 steps; the gradient signal is not converging.
3. **KL constraint near-zero** — observed `kl_loss ≈ 0.001–0.03` × `kl_beta=0.01` ≈ `1e-5`, effectively unconstrained drift from base.
4. **LoRA capacity over-provisioned** — rank=32, alpha=64 (α/r=2), 7 target modules (Q/K/V/O + MLP) gives full-coverage LoRA with aggressive scaling; for 40 noisy steps this invites format-degrading drift.
5. **No intermediate checkpoints** — only the final adapter is saved, so post-hoc "best step" selection is impossible.

v2 addresses (1)–(5) with hyperparam tuning and a single ~6-line trainer change.

## Scope (non-goals)

Out of scope for v2:

- Format-validity reward shaping (preprocess change, separate spec).
- In-loop dev eval / early stopping (requires coupling trainer to vLLM client; revisit if v2 still under-performs).
- Binary variant retrain (`vanilla_full_binary`) — separate run after v2 lands.
- Round-1 → Round-2 iterative rollout refresh — methodology unchanged.

## Files

| Path | Change |
|---|---|
| `grpo/trainer/config/vanilla_full_continuous_v2.yaml` | **NEW** — v2 hyperparams |
| `grpo/trainer/offline_trainer.py` | **MODIFY** — add `save_every_n_steps: int = 0` to `GRPOConfig`; add ~6-line ckpt-save block inside training loop |
| `grpo/trainer/run_grpo.py` | unchanged |

The trainer modification is backward compatible: `save_every_n_steps=0` (default) preserves v1 behavior bit-for-bit. v1 yaml runs identical to before.

## Hyperparameter deltas

| param | v1 | v2 | rationale |
|---|---|---|---|
| `K_steps` | 40 | **178** | 89 batches × 2 epochs → 100% data, 2× exposure |
| `lr` | 3e-6 | **1e-6** | smaller per-step update; combined with α/r=1 below, effective LR ≈ 1e-6 |
| `kl_beta` | 0.01 | **0.05** | 5× stronger drift penalty; β·kl product moves from ~1e-5 to ~1e-4 range |
| `mini_batch_size` | 8 | **16** | gradient variance ~1/√2; halves the noisy sign-flip risk |
| `micro_batch_size` | 1 | 1 | unchanged (mem-safe for 32k context) |
| `precache_micro_batch_size` | 1 | 1 | unchanged |
| `clip_eps` | 0.2 | 0.2 | unchanged (revisit only if v2 still unstable) |
| `grad_clip` | 1.0 | 1.0 | unchanged |
| `lora_rank` | 32 | **16** | halve trainable capacity |
| `lora_alpha` | 64 | **16** | α/r: 2 → 1, conservative scaling |
| `lora_dropout` | 0.0 | 0.0 | unchanged |
| `lora_target_modules` | 7 (Q/K/V/O + gate/up/down) | **3 (Q/V/O)** | attention-only LoRA; MLP modules excluded to constrain expressivity to retrieval/routing, not feature reshaping |
| `save_every_n_steps` | — | **20** | NEW — intermediate ckpts at step 20/40/.../160 |
| `seed` | 42 | 42 | unchanged |

Effective trainable-parameter count drops roughly 4× (rank halved, target modules cut from 7 to 3, α scaled).

## Trainer code change

In `offline_trainer.py`, two edits:

**Edit 1** — add field to `GRPOConfig`:
```python
save_every_n_steps: int = 0  # 0 disables intermediate saves
```

**Edit 2** — inside `run_training`, in the training loop (currently lines ~253–272), after `optimizer.step()` and metrics logging:
```python
if cfg.save_every_n_steps > 0 and (step_iter + 1) % cfg.save_every_n_steps == 0:
    inter_dir = out_dir / f"step_{step_iter + 1}"
    model.save_pretrained(inter_dir)
    logger.info("saved intermediate ckpt → %s", inter_dir)
```

No other changes. Final-save logic at line ~274 is untouched.

## Output layout

`grpo/ckpts/vanilla_full_continuous_v2/round1/`:
```
adapter_config.json       # final (step 178)
adapter_model.safetensors
config.json
metrics.jsonl
step_20/
  adapter_config.json
  adapter_model.safetensors
step_40/
step_60/
...
step_160/
```

8 intermediate checkpoints (steps 20, 40, …, 160) + final (step 178). Each adapter ≈ 66 MB at rank=16 with 3 target modules (v1 was 309 MB at rank=32 × 7 modules; v2 scales by (16/32) × (3/7) ≈ 21%). Total ≈ 9 × 66 MB ≈ 600 MB.

## Evaluation

After training, evaluate each ckpt on `dev.txt` (56 tasks) using the existing eval pipeline (`appworld/experiments/configs/eval/dev/qwen25_7b_lora_vanilla_full_continuous_round1.jsonnet` template — duplicate for each step's LoRA name). Pick the step with the best `task_goal_completion`.

This replaces in-loop early stopping with offline post-hoc selection — same outcome, fraction of the complexity.

## Risks

- **2 epochs may overfit on small data (89 task groups)**: monitor whether step 89 → step 178 metrics diverge in pg_loss / kl. If overfitting visible, single epoch (K=89) is the fallback.
- **MLP exclusion may starve LoRA of capacity to learn long-horizon corrections**: if v2 underperforms v1 on dev despite all other improvements, attention-only is the suspect — add `gate_proj` back as next iteration.
- **Lower lr + stronger KL may produce a "barely-moved" policy**: if dev scores are statistically indistinguishable from base across all 9 ckpts, the constraints are too tight; relax `kl_beta` to 0.03 and `lr` to 2e-6 in v3.

## Verification

Before launching v2 training:
1. Run trainer with v1 yaml — confirm output unchanged (smoke).
2. Run trainer with v2 yaml dry — confirm K=178 wraps cached batches correctly via existing line 233-235 logic; confirm step_20/ directory appears at step 20.

After training:
1. Inspect `metrics.jsonl` — pg_loss/kl_loss trajectory should be visibly smoother than v1.
2. All 9 ckpts present on disk.
3. Run dev eval pipeline for each ckpt; produce a step → dev success-rate table.
