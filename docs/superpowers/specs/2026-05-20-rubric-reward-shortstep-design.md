# Short-step Variant of the Joint Segmentation+Reward Scorer

**Date:** 2026-05-20
**Scope:** `rubric_reward/` — add a second scoring variant that prefers shorter trajectories.

## Motivation

The current joint-rubric prompt (`run_rollouts_KS.py`) scores each segment in isolation and treats a successful recovery as genuinely valuable (4–5). This is consistent with "score each segment on its own outcome", but it leaves a gap in the training signal: a rollout that completes a subgoal cleanly in N steps and one that completes the same subgoal in N + k steps after an earlier failed attempt can receive identical or near-identical mean contributions. For SeRPO we want the advantage signal to also prefer **fewer steps** at fixed correctness, so the policy is pushed toward "get it right the first time" rather than "recover from a hallucinated API call."

We add a second variant that encodes the strict ordering:

> success_short > recovery_success > recovery_fail

without changing the output schema or any downstream consumer.

## Approach

Keep the joint segmentation + 1–5 contribution rubric. Inject an **efficiency penalty** into the existing 1–5 rubric so that:

- A clean, short-path segment earns 5.
- A successful recovery segment (one where a prior segment failed at the same subgoal) is **capped at 4**.
- A failed segment continues to score 1–2 as in the baseline.

The cap is preferred over a per-step subtraction rule because the LLM can reliably detect "did an earlier segment in this trajectory fail at this same subgoal?" but is unreliable at estimating a counterfactual minimum step count for AppWorld tasks.

## Files

- **Rename:** `rubric_reward/run_rollouts_KS.py` → `rubric_reward/run_rollouts_KS_baseline.py` (via `git mv` to preserve history).
- **New:** `rubric_reward/run_rollouts_KS_shortstep.py` (copy of baseline, with the prompt edits and output-path changes below).

No callsites to update — both files are entrypoint scripts.

## Changes inside `run_rollouts_KS_shortstep.py`

### Output paths

Both variants must coexist; the resume logic in `load_already_done()` keys on filename, so the two variants must write to different files or they will silently cross-contaminate on resume.

- Baseline: `results/rollout/joint_<seed>.jsonl`, `results/rollout/pretty/`
- Shortstep: `results/rollout/shortstep_<seed>.jsonl`, `results/rollout/pretty_shortstep/`

### Prompt edits (three surgical additions to `JOINT_SEGMENT_REWARD_PROMPT`)

Everything else — template slots `{instruction}`, `{trajectory}`, `{evaluation_summary}`; the 7 phase types; the 1–5 contribution schema; the output JSON shape; the post-processing pipeline (`validate_segments`, `merge_consecutive_same_score`) — is unchanged.

**Edit 1 — new "Efficiency" anchor inside the 1–5 rubric block:**

> **Efficiency matters within a score band.** A segment that completes its subgoal in fewer steps is strictly better than one that takes more steps. A clean short-path segment earns the top of its band; an inefficient one earns the bottom.

**Edit 2 — new explicit ordering rule in "Anchoring principles":**

> **Successful recovery is capped at 4, not 5.** Score 5 is reserved for segments that achieved their subgoal cleanly on the first attempt with no prior failed segment pursuing the same subgoal. If an earlier segment in the trajectory failed at the same subgoal (e.g., a hallucinated-API crash for the same data lookup), the segment that finally succeeds at that subgoal scores 4 even if its own execution is clean. Rationale: the total trajectory cost includes the wasted prior steps, and we want to reward the policy that gets it right the first time over one that recovers.

**Edit 3 — update the existing AppWorld failure-pattern block so its score band is consistent with the new cap.** Change:

> The subsequent recovery segment that finds the right API and makes progress gets **4-5** — it did real productive work, even if it wouldn't have been needed without the earlier mistake.

to:

> The subsequent recovery segment that finds the right API and makes progress gets **4** (capped because a prior segment failed at the same subgoal) — it did real productive work, even if it wouldn't have been needed without the earlier mistake.

### What is explicitly NOT changed

- The final `completion` segment's score is still graded on evaluation outcome (5 correct, 1 wrong, 3 near-miss), independent of step count. Step-count pressure applies to intermediate subgoals.
- JSON output schema (`start_step`, `end_step`, `subgoal`, `type`, `contribution`, `rationale`).
- The `VALID_TYPES` set, size guards (`MAX_TRAJECTORY_BYTES`, `MAX_STEPS`), retry logic, pricing table.
- Pretty-dump structure.

## Verification

This spec does not require automated tests. After implementation, sanity-check by running the smoke mode on a few seed-1 tasks:

```
python rubric_reward/run_rollouts_KS_shortstep.py --seed 1 --limit 3
```

and visually compare the resulting `pretty_shortstep/seed_1__*.json` against the existing baseline `pretty/seed_1__*.json` for the same tasks. Expectation: on rollouts that contain a hallucinated-API crash followed by a successful recovery, the recovery segment scores 4 in the shortstep variant versus 4–5 in the baseline; clean trajectories should be largely unchanged.

Full 720-rollout runs and downstream SeRPO ablations are out of scope for this spec — they are training-pipeline work tracked separately.

## Out of scope

- Changing the baseline rubric.
- New output fields or schema changes.
- Per-step subtraction rules or counterfactual-step estimation.
- Subgoal-group clustering as a separate output structure.
