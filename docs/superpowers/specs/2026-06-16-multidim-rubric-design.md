# Multidimensional (count-anchored) rubric for SeRPO reward scoring

**Date:** 2026-06-16
**Status:** Design (pre-implementation)
**Scope:** `rubric_reward/run_rollouts_KS_baseline.py` ONLY (new reward-signal variant; other KS_* variants unchanged).

## Goal

Replace the single holistic 1–5 `contribution` score per segment with **4 quantitative, count-anchored dimensions** (each 1–5), averaged into the `contribution` value the RL trainer consumes. Every segment is scored on all 4 dimensions on the same scale; the average is always over 4 comparable numbers. The per-dimension scores are preserved in the output for later analysis/re-weighting.

## Why

The current holistic 1–5 bundles correctness, efficiency, progress, and exploration-discipline into one vibes-based number — noisy and hard to defend ("what's the difference between 3 and 4?"). Count-anchored dimensions tie each score to **observable trajectory facts** (error counts, wasted-step counts, doc-call-before-action), and the judge must cite the count in its rationale, making every score auditable.

## The 4 dimensions (count-anchored)

Each scored 1–5 by the LLM judge, which **counts the signal from the trajectory and states the count in `rationale`**, then maps to the score via fixed thresholds.

### 1. correctness — errors + final outcome
- 5 = 0 errors, ended successfully
- 4 = exactly 1 error, then succeeded
- 3 = 2 errors, then succeeded
- 2 = ≥1 error, ended without success
- 1 = crashed on hallucinated/nonexistent API, or wrong approach, no recovery
- rationale cites: "N errors, ended success/fail"

### 2. efficiency — wasted steps
`wasted = (#steps in segment) − (#steps the subgoal actually needs)`, plus redundant/repeated identical calls.
- 5 = 0 wasted · 4 = 1 · 3 = 2 · 2 = 3–4 · 1 = ≥5 wasted or repeat-loop
- A clean 1-step segment (e.g. a lone login or completion) = 5.
- rationale cites: "K wasted steps (what they were)"

### 3. progress — downstream use
- 5 = produced goal-relevant state that a LATER segment depends on
- 3 = info only partially used / partial progress
- 1 = output unused / dead-end (nothing downstream depends on it)
- rationale cites: "used by step X" or "unused"

### 4. exploration_fit — explore-before-act (targets the 78.5% hallucinated-API failure mode)
- 5 = explored the API (`apis.api_docs.show_*`) before the first state-changing action on a new app, OR correctly reused an already-known API, OR **no exploration was required for this segment type** (login/data_fetch/computation/completion → default 5)
- 3 = acted with partial/incomplete exploration
- 1 = acted on a new app with ZERO prior api_docs lookup (premature-action pattern)
- rationale cites: "explored before acting" / "no doc lookup before action" / "n/a, no exploration needed"

## "Every segment, all 4 dims" + N/A handling

Every segment (any of the 7 phase types) is scored on **all 4 dimensions** — never "N/A," never a skipped field, so the 4-way mean is always comparable across segment types.

A dimension that does not naturally apply to a segment type maps to the **non-penalizing default = 5** ("didn't need to, no harm"). Primary case: `exploration_fit = 5` for login/data_fetch/computation/completion segments (no new-app action to evaluate). This preserves the old holistic intent (a clean login was always a high score) and avoids injecting an arbitrary guess. Decision rationale: minimalism + don't penalize a segment for omitting something irrelevant to its type.

## Aggregation

`contribution = round((correctness + efficiency + progress + exploration_fit) / 4, 3)` — a **fractional** value (e.g. 3.25). Computed **in code** (`validate_segments`), NOT emitted by the judge. The judge emits only the 4 dimension integers.

## Output schema (per segment)

```json
{"start_step": int, "end_step": int, "subgoal": str, "type": str,
 "correctness": 1-5, "efficiency": 1-5, "progress": 1-5, "exploration_fit": 1-5,
 "contribution": <float mean>, "rationale": "<must cite the counts>"}
```

## Worked examples (verify the mean reproduces old-scale intent)

- Hallucinated-API crash: correctness 1, efficiency 2, progress 1, exploration_fit 1 → contribution 1.25 (≈ old "1").
- Clean successful recovery: correctness 5, efficiency 3, progress 5, exploration_fit 5 → contribution 4.5 (≈ old "4–5").
- Clean lone login: correctness 5, efficiency 5, progress 5 (token used downstream), exploration_fit 5 (n/a default) → 5.0.

## Code changes (4 spots, all in `run_rollouts_KS_baseline.py`)

1. **`JOINT_SEGMENT_REWARD_PROMPT`**: replace the "Contribution scoring (1-5)" block with the 4 count-anchored dimensions above (each with anchors + "cite the count" instruction) and the N/A→5 rule. Change the output-format block to emit the 4 dimension fields instead of `contribution`. Segmentation rules, 7 phase types, granularity/merge guidance unchanged.

2. **`validate_segments`**: read & validate the 4 dimension keys (each `int`, 1–5; reject segment if any missing/out-of-range). Compute `contribution = round(sum(4)/4, 3)`. Store all 5 values (4 dims + contribution) plus existing fields in the returned dict. (Currently reads `contribution` directly → flips to computed.)

3. **`merge_consecutive_same_score`**: unchanged logic — merges adjacent segments sharing `type` AND equal (now fractional) `contribution`. Operates on the computed value; works as-is. (Fractional means fewer spurious merges — acceptable/good.)

4. **Record fields / pretty output**: `mean_contribution` / `min_contribution` / `max_contribution` already derive from `contribution`; keep working over fractional values. The 4 per-dim fields ride along in each segment dict automatically into the JSONL and pretty output.

## What does NOT change

- Segmentation logic, 7 phase types, joint single-call design, GPT-5-mini, retries/backoff, ThreadPool parallelism, resume-by-task-id, all paths.
- **Downstream RL** (`grpo/preprocess/make_advantages.py: compute_serpo_advantage`) reads `seg["contribution"]` — unchanged; now receives a fractional mean instead of an integer. Advantages are float, so no trainer change.
- The other 3 KS_* variants (`_shortstep`, `_giveup`, `_noerror`) keep the holistic single-score rubric. This is a NEW reward-signal variant.

## Out of scope / caveats

- **Re-scoring round0_9b with this rubric is a separate, paid GPT-5-mini run** — must be approved before running. The code change alone produces no new data.
- More output tokens per segment (4 scores + count citations) → modestly higher cost/latency vs the holistic score.
- Output variant directory stays `KS_baseline` per the existing script; if a clean side-by-side with the old holistic scores is wanted, point this at a NEW variant dir (e.g. `KS_multidim`) so resume-by-task-id doesn't cross-contaminate. (Decision deferred to implementation/plan.)
