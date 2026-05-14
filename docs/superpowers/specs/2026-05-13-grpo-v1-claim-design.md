# GRPO Trainer (Phase 3) — v1-claim Design

- Status: Draft (awaiting user review)
- Date: 2026-05-13
- Project root: `/data/minjeong/Autonomous_agent/`
- Working subtree: `grpo/`

## 1. Context

Phase 3 of the SeRPO (Segment-Reward Policy Optimization) research pipeline. Phase 0–2 already produced:

- **Phase 0**: External vLLM serving Qwen2.5-7B-Instruct with LoRA hot-swap (`serve_qwen_vllm.sh`, `--enable-lora --max-loras 4 --max-lora-rank 64`).
- **Phase 1**: Round-0 rollout pool — 8 seeds × 90 tasks = 720 trajectories at `appworld/experiments/outputs/rollout/round0/seed_{1..8}/tasks/<tid>/logs/lm_calls.jsonl`. Per-seed binary outcome labels in `evaluations/train.json`.
- **Phase 2**: Joint segmentation + per-segment 1-5 contribution scoring (gpt-5-mini, `rubric_reward/run_rollouts.py`). Output: `rubric_reward/results/rollout/joint_seed_N.jsonl` × 8 (pending execution as of design date). Output schema each row: `{task_id, trajectory, segments: [{start_step, end_step, type, contribution, ...}], evaluation_summary, ...}`.

Phase 3 trains a LoRA adapter via GRPO on this pool.

## 2. Goal & Non-goals

### Goal (v1-claim deliverable)

Run two Round-1 GRPO training experiments on the Fail-Only condition and measure test-set success rate:

1. **Vanilla GRPO** — outcome-only advantage (`Â = z-score(task_goal_completion)`), trajectory-uniform broadcast.
2. **SeRPO** — per-segment contribution advantage (`Â = z-score(contribution)`), per-segment broadcast to that segment's assistant tokens.

Both use the *same* trainer core (verl-based, clipped GRPO + KL-to-ref); they differ only in the advantage tensor produced by `preprocess/make_advantages.py`. After training, both LoRAs are evaluated on `test_normal` (full split) via vLLM hot-swap, and success-rate deltas vs base policy are recorded.

Empirical claim under test: **In the Fail-Only condition where outcome variance ≈ 0, Vanilla GRPO produces near-zero policy gradient while SeRPO produces non-zero gradient via per-segment differentiation — and this is reflected in measurable success-rate improvement on test_normal.**

### Non-goals (deferred to v2+)

- GiGPO baseline
- Full-condition training set (currently only Fail-Only is in v1)
- R-round iterative GRPO orchestration (Round 2+)
- Subgoal-string or type+order segment matching across rollouts (no-matching used in v1)
- Held-out judge model evaluation
- test_challenge split, recovery-rate metric
- Online GRPO (verl rollout worker integration)

## 3. Architecture

verl-based GRPO trainer running on 2× RTX PRO 6000 Blackwell (96GB each). External vLLM remains responsible for rollout serving and post-train LoRA hot-swap evaluation; the trainer is *offline-only* with respect to rollout (no in-process rollout worker for v1).

### 3.1 Data flow

```
inputs
├─ appworld/.../seed_N/tasks/<tid>/logs/lm_calls.jsonl    ─┐
├─ appworld/.../seed_N/evaluations/train.json (outcomes)  ─┤
└─ rubric_reward/results/rollout/joint_seed_N.jsonl       ─┘
                                                           │
                                  grpo/preprocess/build_dataset.py
                                                           │
                                                           ▼
                                  grpo/data/round0/
                                    ├─ failonly_task_ids.json   (81 task_ids)
                                    ├─ vanilla_failonly.parquet (648 rows = 81 × 8)
                                    └─ serpo_failonly.parquet   (648 rows, advantage tensors differ)
                                                           │
                                                           ▼
                                  grpo/trainer/run_grpo.py  (verl recipe + entrypoint)
                                                           │
                                                           ▼
                                  grpo/ckpts/{vanilla,serpo}/round1/
                                                           │
                                                           ▼  (LoRA → vLLM via --lora-modules)
                                  grpo/eval/run_test_normal.py
                                                           │
                                                           ▼
                                  grpo/metrics/{vanilla,serpo}_round1.json
```

### 3.2 Directory layout

```
grpo/
├── preprocess/
│   ├── build_dataset.py        # CLI orchestrator: rollouts + reward → parquet
│   ├── tokenize_trajectory.py  # cumulative lm_calls.jsonl → input_ids + per-step token ranges
│   ├── make_advantages.py      # (method, segments | outcome) → per-token advantage tensor
│   └── failset.py              # 8 seeds × 90 tasks → strict 0/8 fail-only task_ids
├── data/                       # build artifacts (gitignore)
│   └── round0/
│       ├── failonly_task_ids.json
│       ├── vanilla_failonly.parquet
│       └── serpo_failonly.parquet
├── trainer/
│   ├── config/
│   │   ├── vanilla_round1.yaml
│   │   └── serpo_round1.yaml
│   └── run_grpo.py             # verl GRPO entrypoint (shared)
├── eval/
│   └── run_test_normal.py      # LoRA → vLLM hot-swap → appworld test_normal → success rate
├── scripts/
│   ├── train_vanilla_round1.sh
│   └── train_serpo_round1.sh
├── ckpts/                      # LoRA checkpoints (gitignore)
└── metrics/                    # eval outputs (gitignore)
```

## 4. Components

### 4.1 `preprocess/tokenize_trajectory.py`

**Input.** Path to `seed_N/tasks/<tid>/logs/lm_calls.jsonl`.

**Operation.**

1. Read the **last** line of `lm_calls.jsonl`, extract `input.messages` (cumulative; contains all prior turns).
2. Locate the user message ending with the substring `"Using these APIs, now generate code to solve the actual task:"`. Everything before and including this message is the **prefix** (system + few-shot demos + task instruction). Messages after this user message are the actual interaction.
3. The remainder is structured as alternating `(assistant, user)` pairs: step 1 = (assistant₁, user₁), step 2 = (assistant₂, user₂), …, step N = (assistant_N, …). The final step may not have a trailing user message.
4. Apply Qwen2.5 chat template via `tokenizer.apply_chat_template(messages, return_offsets_mapping=True, return_dict=True)` to obtain `input_ids` and per-message character offset spans.
5. For each step k, record the token range `[start_tok_k, end_tok_k)` covering its **assistant** message.

**Output.**

```python
{
  "task_id": str,
  "seed": int,
  "input_ids":      List[int],          # full sequence (system + few-shot + task + all turns)
  "attention_mask": List[int],          # all 1 (no padding at this stage)
  "response_mask":  List[int],          # 1 = assistant token (steps 1..N), 0 = everything else
  "step_token_ranges": List[Tuple[int,int]],  # length N, step_token_ranges[k-1] = (start_tok_k, end_tok_k)
}
```

**Mask rule (definitive).**

| Token span | `response_mask` |
|---|---|
| System prompt | 0 |
| Few-shot demo messages (user *and* assistant) | 0 |
| Task instruction user message | 0 |
| Step k assistant message (k = 1..N) | **1** |
| Step k env-output user message | 0 |

### 4.2 `preprocess/failset.py`

**Input.** All 8 `seed_N/evaluations/train.json`.

**Operation.** For each `task_id`, count seeds where `individual[task_id]["success"] == True`. Emit task_ids where this count is 0 (strict 0/8).

**Output.** `grpo/data/round0/failonly_task_ids.json` — JSON list of strings. Expected size: **81** (verified empirically 2026-05-13).

### 4.3 `preprocess/make_advantages.py`

**Input.** For a single task_id, the 8 per-seed tokenized records + per-seed `joint_seed_N.jsonl` segment data + per-seed binary outcomes.

**Vanilla GRPO advantage.**

```python
def compute_vanilla(rollouts):  # 8 rollouts of one task
    outcomes = [r.outcome for r in rollouts]          # all 0 in fail-only set
    mu = mean(outcomes); sigma = std(outcomes) + 1e-8
    for r in rollouts:
        scalar = (r.outcome - mu) / sigma             # = 0 when sigma == 1e-8 numerator
        r.token_adv = np.zeros_like(r.input_ids, dtype=float32)
        r.token_adv[r.response_mask == 1] = scalar    # trajectory-uniform broadcast
```

In strict fail-only (all outcomes = 0), `mu = 0, sigma = 1e-8`, so `scalar = 0`. This is the intended behavior — Vanilla GRPO has no signal in this regime; the experiment demonstrates it.

**SeRPO advantage (no matching).**

```python
def compute_serpo(rollouts):
    pool = [seg.contribution for r in rollouts for seg in r.segments]
    mu = mean(pool); sigma = std(pool) + 1e-8
    for r in rollouts:
        r.token_adv = np.zeros_like(r.input_ids, dtype=float32)
        for seg in r.segments:
            a_hat = (seg.contribution - mu) / sigma
            start_step, end_step = seg.start_step, seg.end_step
            for step_k in range(start_step, end_step + 1):
                tok_start, tok_end = r.step_token_ranges[step_k - 1]
                r.token_adv[tok_start:tok_end] = a_hat
        # tokens in segmentation gaps remain 0
```

Segments are taken post-merge from `joint_seed_N.jsonl[*].segments`. Steps outside any segment retain `token_adv = 0` (still receive `response_mask=1`, so they're in the loss denominator but contribute zero numerator).

### 4.4 `preprocess/build_dataset.py`

**CLI.** `python -m grpo.preprocess.build_dataset --method {vanilla,serpo} --condition failonly --output-dir grpo/data/round0/`

**Operation.**

1. Load `failonly_task_ids.json` (build if missing).
2. For each task_id × seed:
   - Tokenize trajectory.
   - Apply skip rules (see §6).
   - Attach raw segments + outcome.
3. Group by task_id (8 rollouts per group), call `compute_<method>` to populate `token_adv`.
4. Write parquet with rows shaped as:

```python
{
  "task_id": str, "seed": int,
  "input_ids": List[int], "attention_mask": List[int],
  "response_mask": List[int], "advantages": List[float],
  "outcome": int,                          # for inspection
  "step_token_ranges": List[List[int]],    # for debug
  "raw_segments": List[Dict],              # for debug
}
```

### 4.5 `trainer/run_grpo.py` + `trainer/config/*.yaml`

verl GRPO recipe with the following customizations:

- **Dataset.** Custom `RLHFDataset`-equivalent that reads the parquet directly. Per-row fields → DataProto fields: `input_ids, attention_mask, response_mask` and `advantages` (precomputed; bypass verl's reward worker entirely).
- **Reward / advantage computation.** Disabled. We do not use verl's `reward_fn` or `compute_advantage`. The advantage tensor in the parquet is the final per-token advantage; verl's GRPO loss reads it directly.
- **Actor.** Qwen/Qwen2.5-7B-Instruct + peft LoRA (`r=32, α=64, target=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj, dropout=0`), bf16, FSDP.
- **Old policy.** Snapshot at round start (= base Qwen + currently 0 LoRA delta for Round 1). `old_logp` computed once before the K gradient steps.
- **Ref policy.** Same model with `peft.disable_adapter()` context — no second model instance. `ref_logp` computed once before training.
- **KL.** Schulman k3 estimator: `kl = exp(ref_logp − new_logp) − (ref_logp − new_logp) − 1`. Always ≥ 0.
- **Loss.**

```
ratio = exp(new_logp − old_logp)
pg_loss = − mean( response_mask · min(ratio · adv, clip(ratio, 1−ε, 1+ε) · adv) )
kl_loss =   mean( response_mask · k3(ref_logp, new_logp) )
loss    = pg_loss + β · kl_loss
```

The two yaml configs (`vanilla_round1.yaml`, `serpo_round1.yaml`) differ **only** in the dataset path (`vanilla_failonly.parquet` vs `serpo_failonly.parquet`) and output checkpoint dir. All training hyperparameters are identical.

### 4.6 `eval/run_test_normal.py`

1. Take a LoRA checkpoint dir.
2. Push it to the running external vLLM via its `--lora-modules` registry (hot-swap; vLLM was launched with `--enable-lora --max-loras 4`).
3. Invoke `appworld run test_normal --model-name qwen25_7b_lora_<method> --agent-name simplified_react_code_agent --dataset-name test_normal --num-processes 4` with config pointing to the LoRA-named model server.
4. Parse `evaluations/test_normal.json` → aggregate `task_goal_completion` (success rate).
5. Write `grpo/metrics/<method>_round1.json` with: success_rate, base_policy_success_rate, delta, n_tasks, per-task results.

Eval temperature = 0.7, single sample per task (G_eval = 1).

### 4.7 `scripts/train_{vanilla,serpo}_round1.sh`

Sequence:
1. `python -m grpo.preprocess.build_dataset --method <method> --condition failonly`
2. `python -m grpo.trainer.run_grpo --config grpo/trainer/config/<method>_round1.yaml`
3. `python -m grpo.eval.run_test_normal --ckpt grpo/ckpts/<method>/round1 --output grpo/metrics/<method>_round1.json`

## 5. Algorithm reference — full GRPO update

For each optimizer step within Round 1:

1. Draw 8 trajectories (mini-batch) from the parquet, processed as 8 micro-batches of 1 trajectory each (gradient accumulation = 8). Sampling is a shuffled draw without replacement across the round; ordering does not affect correctness because advantage tensors are precomputed per-task offline.
2. For each trajectory:
   - Pre-cached at round start: `old_logp[T]` (= initial actor logp, computed once with LoRA loaded), `ref_logp[T]` (= base-model logp via `peft.disable_adapter()`, computed once), `advantages[T]`, `response_mask[T]`.
   - Forward through current actor LoRA → `new_logp[T]`.
3. Compute `ratio`, `pg_loss`, `kl_loss` as in §4.5. Mask by `response_mask`. Mean over masked positions.
4. Backward, AdamW step, gradient clip @ 1.0.

K = 20 such optimizer steps per round (≈ 160 trajectories sampled out of 648 available). After K steps: write LoRA adapter to `grpo/ckpts/<method>/round1/`.

## 6. Error handling & filtering

| Case | Action |
|---|---|
| `lm_calls.jsonl` missing or empty | Skip, log to `preprocess/skipped.jsonl` |
| Tokenized length > 32000 (vLLM max_model_len) | Skip, log |
| Prefix marker `"Using these APIs, now generate code to solve the actual task:"` not found | Skip, log (indicates malformed rollout) |
| `joint_seed_N.jsonl` does not contain task_id (Phase 2 skipped/failed it) | Skip, log |
| Segment `[start_step, end_step]` references step number > N | Clip to N, log warning |
| Steps in a trajectory not covered by any segment | `token_adv = 0` for those tokens (already zero-initialized) |
| Task_id has fewer than 8 valid rollouts after filtering | Skip the whole task group, log (preserves group-stat validity) |
| LoRA hot-swap to vLLM fails | Eval step fails fast; report. Manual investigation required (no auto-retry). |

## 7. Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | |
| LoRA rank `r` | 32 | |
| LoRA alpha | 64 | `α = 2r` |
| LoRA dropout | 0 | |
| LoRA targets | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | |
| Precision | bf16 | |
| Optimizer | AdamW | |
| Learning rate | 1e-6 | |
| Gradient clip | 1.0 | |
| Clip ε | 0.2 | GRPO standard |
| KL coefficient β | 0.01 | |
| KL estimator | k3 (Schulman) | Always ≥ 0 |
| Mini-batch (trajectories per optimizer step) | 8 | |
| Micro-batch | 1 trajectory | gradient accumulation = 8 |
| K (optimizer steps per round) | 20 | ≈ 25% of one epoch (648/8 = 81 steps/epoch); kept small to avoid over-fitting on the 81-task fail-only pool |
| Sampling within round | shuffled draw without replacement, padded by re-shuffling if K × 8 > 648 | advantage tensors are precomputed per-task, so any shuffle order preserves group statistics |
| Max sequence length | 32000 | matches vLLM `max-model-len` |
| FSDP sharding | full | 2× 96GB GPUs |
| Gradient checkpointing | on | required for 32k context |

## 8. Testing strategy

### 8.1 Unit tests

| Module | Test |
|---|---|
| `tokenize_trajectory` | Given a known sample task: (a) prefix marker found at correct index, (b) `response_mask` 1-ratio in expected range (≈ 5–15% for AppWorld trajectories), (c) `step_token_ranges` count matches step count in `joint_seed_N.jsonl`, (d) decoding each step's tokens yields the original assistant text. |
| `make_advantages.compute_vanilla` | 8 mock rollouts all `outcome=0` → all `token_adv = 0`. 8 rollouts with mixed outcomes → group mean ≈ 0, variance > 0. Mask invariants: `token_adv != 0` only where `response_mask == 1`. |
| `make_advantages.compute_serpo` | Group mean of `token_adv[response_mask == 1]` ≈ 0 (within numerical tolerance) when summed across all 8 rollouts of a task. Segment-boundary tokens show `token_adv` changes. Tokens in segmentation gaps stay 0. |
| `failset` | 90 mock tasks with known success counts → only those with `count == 0` returned. Empirical test on real data: returns exactly 81. |

### 8.2 Integration test

End-to-end on 1 task × 8 seeds:
1. Run `build_dataset` for both methods.
2. Load parquet, run 1 forward + backward pass.
3. Assert: loss finite, KL ≥ 0, at least one LoRA parameter has non-zero gradient.

### 8.3 Smoke training run

K = 2 gradient steps on the full 81-task parquet:
- No NaN/Inf in loss or grads.
- LoRA adapter saves and reloads correctly.
- vLLM hot-swap accepts the adapter without error.

### 8.4 Evaluation sanity

After full Round-1 training:
- Vanilla GRPO LoRA on test_normal: success rate should be **statistically indistinguishable from base policy** (because advantage was identically zero in fail-only). If a clear deviation appears, investigate KL pressure or numerical artifacts.
- SeRPO LoRA on test_normal: success rate should be **≥ base policy** (the directional claim). Magnitude is the research finding to report.

## 9. Open questions (out of scope for v1, recorded for v2)

- **Cross-rollout segment matching.** v1 uses no-matching pooled normalization. v2 ablation: subgoal-string fuzzy matching vs no-matching, on the same Round-1 setup.
- **Round-refresh orchestration.** R-round iterative GRPO requires LoRA → vLLM hot-swap → re-rollout → re-process Phase 2. Bash-driven; defer until v1 results justify continuation.
- **GiGPO baseline.** Same trainer + a different advantage function (prefix-aware grouping). Add when needed for the 3-method comparison table.
- **Full-condition training set.** All 90 tasks (not just 81 fail-only). Useful for the realistic-deployment framing (Frame B Table 2).
- **Held-out judge.** Train with gpt-5-mini scores; evaluate with a different model family (Claude/Gemini) on test_normal.

## 10. Cross-references

- `[[project_methodology_iterative_grpo]]` — Frame B framing, GRPO clip + KL choice rationale.
- `[[project_pipeline_state]]` — Phase 2 status (joint reward execution pending).
- `[[project_joint_reward_pipeline]]` — input data format (segments + contribution).
- `[[project_appworld_failure_pattern]]` — 78.5% of failures = hallucinated APIs; motivates per-segment differentiation.
- `[[project_compute_environment]]` — 2× RTX PRO 6000 Blackwell 96GB.
