# GiGPO Offline Baseline — Design

**Date:** 2026-06-09
**Status:** Design (pre-implementation)
**Reference:** GiGPO — *Group-in-Group Policy Optimization for LLM Agent Training* ([arXiv 2505.10978](https://arxiv.org/abs/2505.10978), NeurIPS 2025). Authoritative source: the arXiv v3 TeX (`gigpo/texsrc/body/method.tex`, `.../experiment.tex`, `.../appendix.tex`) — all equations and hyperparameters below are taken verbatim from it. Reference implementation: `gigpo/core_gigpo.py` (verl-agent, langfengQ/verl-agent @ master).

## 1. Purpose & positioning

GiGPO is the **outcome-reward, step-granularity** baseline that completes the 2×2 against SeRPO (judge-reward, segment-granularity):

| | Outcome reward (binary) | Judge reward (rubric 1–5) |
|---|---|---|
| **Trajectory granularity** | vanilla GRPO | serpo_avg |
| **Finer granularity** | **GiGPO** (this spec) | serpo |

It isolates the variable the paper rests on: *does the LLM-judge semantic-phase signal beat free binary-outcome anchor-state grouping?* GiGPO never calls the judge, never sees a segment or a 1–5 score — its entire signal is AppWorld's `success` field (0/1) grouped by repeated environment state.

**Implemented as a fourth advantage method in the existing offline preprocessing pipeline** — no trainer changes, no verl dependency. The reference `core_gigpo.py` is coupled to verl's online per-step `DataProto`; we reimplement the *algorithm* (Eqs. 2–8) in our offline one-row-per-trajectory model, exactly as `make_advantages.py` already does for vanilla/serpo/serpo_avg.

### Empirical viability (confirmed)

Anchor states repeat on AppWorld. Across 20 tasks / 5,310 steps (8 seeds each):
- **Exact-match grouping:** 74% of steps in a group of ≥2, avg group size 2.84.
- **Similarity-0.9 grouping:** ~78% grouped.

Step-level advantage is therefore not degenerate — GiGPO is a real baseline, not a strawman.

## 2. Decisions (locked)

| Decision | Value | Source |
|---|---|---|
| Episode reward (for `A^E`) | **binary** outcome (`success` 0/1) | makes GiGPO the no-judge baseline. See reward-model note below — this is a simplification of the paper, not paper-native |
| Step return (for `A^S`) | `R_k = γ^(N−k)·o` (terminal-only, discounted) | **AppWorld-offline simplification**, NOT paper-faithful. See reward-model note below |
| Condition | **`full` only** | binary on fail-only → all-zeros → degenerate; hard-error if requested |
| Composition | additive `A = A^E + ω·A^S` | paper Eq. 8 |
| ω (weight) | **1.0**, configurable | paper: "fixed at 1 without further tuning" |
| γ (discount) | **0.95**, configurable | paper: γ=0.95 across all benchmark configs (Eq. 5) |
| Normalization | **`leave_one_out`** (subtract group mean, no ÷std) default; `std` available | paper: F_norm=1 = rescaled RLOO; safest for long-horizon/imbalanced groups (our regime). **But minor** — see note below |
| Similarity threshold | **0.9**, off by default | paper E.1 |

**Reward-model gap vs. the paper (important — do not call our reward "paper-native"):** The paper's *actual* experiments (appendix.tex §E.1) are **not pure terminal-reward**. ALFWorld/WebShop use success reward = **10** (not 1) and a **per-step invalid-action penalty of −0.1**; Search-QA uses success = 1 with −0.01 invalid penalty. Therefore the paper's step discounted return `R_t = Σ_{k=t}^T γ^{k-t} r_k` (Eq. 5) **includes intermediate penalty terms** — it is not `γ^{T-t}·o`.

Two consequences for this spec:
1. **Episode vs. step must be stated separately.** The "R(τ)=1 success / 0 failure" phrasing belongs to the **episode-level** discussion around Eq. 2 (and even there the real experiment uses 10, not 1). It must NOT be read as describing the **step-level** discounted return. Our table now lists the two levels on separate rows.
2. **Our `R_k = γ^(N−k)·o` is a deliberate AppWorld-offline simplification, NOT paper-faithful.** The frozen AppWorld rollouts give us a binary terminal outcome and no per-step invalid-action reward signal, so the step return reduces to a discounted terminal reward. This is the same simplification already flagged in §6 as an "AppWorld-specific limitation"; the table rows and this note make explicit that it applies to the **step return**, not just the episode reward.
3. **Reward scale (10 vs 1) is irrelevant.** It is a positive constant multiplier on all advantages → absorbed into the learning rate (same argument as the F_norm=1 / RLOO rescaling), so it does not change gradient direction. We use 0/1; not worth matching, but stated so a reader does not mistake 0/1 for paper fidelity.

**γ is load-bearing:** with terminal-only reward and γ=1, every step in a successful trajectory gets identical return → step advantage collapses toward the episode advantage (redundant). γ<1 (Fig. 3) is what differentiates steps by distance-to-reward. Default must be 0.95, not 1.0. Paper uses γ=0.95 across all benchmark configs.

**F_norm is a minor knob (paper, exp. §):** the paper is explicit that F_norm is "task-dependent rather than universally helpful" and that the w/std vs w/o-std gap is "comparatively minor compared to structural ablations." The *structural* parts (having both A^E and A^S) are the real driver. We default to `leave_one_out` because it is the safer choice for imbalanced/long-horizon groups (AppWorld), not because it is load-bearing. Reporting both variants is optional, not required.

**What the GiGPO step signal actually tests on AppWorld (paper §exp, Fig. 3):** anchor-state grouping specifically suppresses repetitive/redundant action loops (`call→fail→call→fail…`) by pooling repeated states and ranking the actions taken from them. This is directly relevant to the AppWorld failure mode (looping on failed/hallucinated API calls). So the GiGPO-vs-SeRPO contrast tests *loop-suppression via state-grouping* (GiGPO) vs. *phase-quality via judge* (SeRPO) — a sharper framing than generic "finer credit."

## 3. Architecture

```
appworld/.../rollout/round0_9b/seed_{1..8}/tasks/<tid>/logs/lm_calls.jsonl
   │
   ├─ tokenize_trajectory()  → input_ids, response_mask, step_token_ranges,
   │                            step_anchor_obs   ← NEW additive output
   ▼
compute_gigpo_advantage(rollouts)   ← NEW in make_advantages.py
   │   A^E = loo_norm(binary outcome, across 8-rollout group)        [uniform broadcast]
   │   A^S = loo_norm(discounted step return, within anchor cluster) [per-step broadcast]
   │   token_adv = A^E + ω·A^S
   ▼
build_dataset.py  (method="gigpo")  →  gigpo_full_binary.parquet
   ▼
offline_trainer  (UNCHANGED)  →  LoRA ckpt
```

## 4. Components

### 4a. `extract_anchor_obs` — additive output of `tokenize_trajectory.py`

Returns one anchor string per assistant step = the env-output user message (`Output:` block) immediately preceding that step. Reuses the message-walk already done for `step_token_ranges`. First step's anchor = the shared task-instruction prefix. Returned as `step_anchor_obs: list[str]` (length = num_steps). Other methods ignore the new field; no behavior change for vanilla/serpo/serpo_avg.

### 4b. `compute_gigpo_advantage(rollouts, *, gamma=0.95, omega=1.0, norm_mode="leave_one_out", enable_similarity=False, similarity_thresh=0.9)` — new in `make_advantages.py`

Operates on the 8-rollout task group (same contract as existing advantage functions), mutates `r.token_adv`. Faithful to Eqs. 2–8:

1. **Episode advantage `A^E`** (Eq. 3) — group the 8 binary outcomes, subtract group mean (`leave_one_out`); broadcast uniformly onto each rollout's assistant tokens. Identical to `vanilla` with `remove_std=True`.
2. **Anchor-state grouping** (Eqs. 4, 6) — pool all steps across the 8 rollouts of the task; cluster by `to_hashable(anchor_obs)` (exact), or greedy `SequenceMatcher ≥ thresh` if `enable_similarity`. Hashmap-based, offline, no extra rollouts.
3. **Discounted step return** (Eq. 5) — terminal-only reward, so step *k* of a trajectory with N steps and outcome `o`: `R_k = γ^(N−k) · o`.
4. **Step advantage `A^S`** (Eq. 7) — within each anchor cluster, subtract cluster mean (`leave_one_out`); broadcast each step's value onto its `step_token_ranges` span.
5. **Joint** (Eq. 8) — `token_adv = A^E + ω·A^S` per token; non-assistant tokens stay 0.

### 4c. Helpers ported from reference (Apache-2.0, attributed in docstring)

`to_hashable` (verbatim), `are_similar` (verbatim). Clustering loop adapted to operate on `(rollout, step)` tuples instead of verl `DataProto` rows.

## 5. Plumbing changes

- **`build_dataset.py`:** add `"gigpo"` to `--method` choices + dispatch branch ([build_dataset.py:157](../../../grpo/preprocess/build_dataset.py)); add args `--gamma`, `--omega`, `--gigpo-norm-mode {leave_one_out,std}`, `--enable-similarity`, `--similarity-thresh` (threaded only into the gigpo branch). **Hard error if `method=gigpo` and `condition=failonly`.** Output filename: `gigpo_full_binary.parquet`. The gigpo branch reads `step_anchor_obs` from the tokenizer result and stores it on `RolloutData` (new field, defaults to `[]` for other methods).
- **`RolloutData`** (`make_advantages.py`): add `step_anchor_obs: list[str] = field(default_factory=list)`. Used only by gigpo.
- **New config** `grpo/trainer/config/gigpo_9b_full_binary_v3i.yaml` — cloned from the vanilla full-binary v3i config (same lr/K/guards/LoRA target modules); only `parquet_path` + `output_dir` + `wandb_run_name` differ.
- **New script** `grpo/scripts/train_gigpo_9b_full_binary_v3i.sh` — clone of the vanilla full-binary train script.

## 6. Error handling / edge cases

- **All-fail group** (defensive; shouldn't occur on `full`): `A^E`=0 and all step returns 0 → `A^S`=0 → all-zero advantage, no NaN.
- **Singleton anchor cluster** (size 1): cluster mean = the value → advantage 0. Matches reference's size-1 handling.
- **Step with empty/missing anchor:** anchor = `""`, hashes consistently into its own/shared empty-string cluster; logged at debug.
- **Segment/step index out of range:** skip silently (same guard as `serpo`).
- **Normalization-scale caveat (documented, not fixed):** `leave_one_out` is not unit-variance, so GiGPO advantages enter the trainer at a different scale than z-scored serpo/vanilla. The trainer's `[-5, 5]` advantage clamp ([offline_trainer.py:356](../../../grpo/trainer/offline_trainer.py)) bounds it. This is the noted normalization caveat in the SeRPO-vs-GiGPO comparison; per the paper, F_norm=1 scaling is absorbable into LR, so it does not bias the gradient direction.
- **AppWorld-specific limitation (documented; see also the reward-model gap note in §2):** our step return is terminal-only (`R_k = γ^(N−k)·o`), so `A^S` differentiation comes entirely from γ-discounting × cluster membership ("among steps hitting this same env-state, which sat closer to an eventual success"). The paper's real step return additionally carries per-step invalid-action penalties (−0.1), which our frozen binary rollouts do not provide; this makes our `A^S` a strictly weaker (terminal-only) instance of the paper's signal. Flagged as a known limitation of the baseline on this benchmark.

## 7. Testing (`grpo/tests/test_make_advantages.py`, extending existing)

- Hand-built 2-rollout group: assert exact `A^E + ω·A^S` token values.
- Anchor grouping: 3 steps, 2 sharing an anchor → correct 2-cluster partition.
- Discounted return: γ=0.95, known step count → returns-to-go match `γ^(N−k)·o`.
- All-fail group → all-zero advantage (no NaN).
- Singleton cluster → step advantage 0.
- `enable_similarity`: anchors differing by one char cluster together at 0.9.
- `norm_mode="std"` path: divides by group std (parity with reference `mean_std_norm`).

## 8. Out of scope (YAGNI / future specs)

- Trainer, dataset loader, the other three advantage functions, the reward pipeline — untouched.
- Any GPU run (training). Will ask before any GPU-occupying command, per the GPU-permission rule.
- **Continuous-reward GiGPO** and the **SeRPO-additive arm** — separate future specs, not here.
