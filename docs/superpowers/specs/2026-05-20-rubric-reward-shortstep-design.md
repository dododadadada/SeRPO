# Short-step and Giveup Variants of the Joint Segmentation+Reward Scorer

**Date:** 2026-05-20
**Scope:** `rubric_reward/` — add two scoring variants alongside the baseline: one that prefers shorter trajectories (`_shortstep`), and one that truncates the trajectory at abandoned-subgoal failures (`_giveup`).

## Motivation

The current joint-rubric prompt (`run_rollouts_KS.py`) scores each segment in isolation and treats a successful recovery as genuinely valuable (4–5). This is consistent with "score each segment on its own outcome", but it leaves two gaps in the training signal for SeRPO:

1. **No short-step preference.** A rollout that completes a subgoal cleanly in N steps and one that completes the same subgoal in N + k steps after an earlier failed attempt can receive identical or near-identical mean contributions. We want the advantage signal to also prefer **fewer steps** at fixed correctness, so the policy is pushed toward "get it right the first time" rather than "recover from a hallucinated API call."
2. **Wasted steps after an abandoned subgoal pollute the training signal.** If the agent fails at a subgoal and moves on to a different subgoal without ever completing the failed one, the downstream steps are working on a different problem — yet they still contribute to the per-rollout mean and bias the advantage. For abandoned-subgoal cases we want to truncate the trajectory at the point of abandonment so the remaining work doesn't bias the signal.

We add **two independent variants** so the effect of each can be measured in isolation:

- `_shortstep`: changes the rubric to encode `success_short > recovery_success > recovery_fail`.
- `_giveup`: keeps the baseline rubric and post-processes the saved record to truncate at abandoned-subgoal failures.

Neither variant changes the output JSON schema or any downstream consumer.

## Approach

Two independent changes, each in its own file. Both keep the joint segmentation + 1–5 contribution rubric and the existing output JSON schema.

### `_shortstep`: efficiency penalty in the rubric

Inject an **efficiency penalty** into the existing 1–5 rubric so that:

- A clean, short-path segment earns 5.
- A successful recovery segment (one where a prior segment failed at the same subgoal) is **capped at 4**.
- A failed segment continues to score 1–2 as in the baseline.

The cap is preferred over a per-step subtraction rule because the LLM can reliably detect "did an earlier segment in this trajectory fail at this same subgoal?" but is unreliable at estimating a counterfactual minimum step count for AppWorld tasks.

### `_giveup`: post-processing truncation at abandoned subgoals

Keep the baseline rubric prompt unchanged. After the model returns segments for the full trajectory, walk the segments and **truncate at the first abandoned-subgoal failure**:

- A segment is "failed" if its `contribution` ≤ 2.
- A failed segment is "abandoned" if the next segment has a different `subgoal` AND different `type` (i.e., the agent moved on rather than retrying).
- When this pattern is found, drop the failed segment and every segment after it. Re-derive `trajectory_text` and `num_steps` from the kept-segment range so downstream consumers see a consistent record.
- A failed segment followed by a same-subgoal recovery attempt is **kept** — that's a legitimate retry and is what `_shortstep` is designed to score.
- If no abandoned-failure pattern exists, the record is identical to baseline.

The model scores the full untruncated trajectory (it needs the full context to assign correct scores). Truncation is applied in Python, is deterministic, and is reflected only in the saved JSON.

## Files

- **Rename:** `rubric_reward/run_rollouts_KS.py` → `rubric_reward/run_rollouts_KS_baseline.py` (via `git mv` to preserve history).
- **New:** `rubric_reward/run_rollouts_KS_shortstep.py` (copy of baseline, with the prompt edits and output-path changes below).
- **New:** `rubric_reward/run_rollouts_KS_giveup.py` (copy of baseline, baseline rubric unchanged, with the truncation post-processor and output-path change below).

No callsites to update — all three are entrypoint scripts.

## Output paths

All three variants must coexist; `load_already_done()` keys on filename, so the variants must write to different files or they will silently cross-contaminate on resume.

- Baseline: `results/rollout/joint_<seed>.jsonl`, `results/rollout/pretty/`
- Shortstep: `results/rollout/shortstep_<seed>.jsonl`, `results/rollout/pretty_shortstep/`
- Giveup:   `results/rollout/giveup_<seed>.jsonl`,    `results/rollout/pretty_giveup/`

## Changes inside `run_rollouts_KS_shortstep.py`

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

### What is explicitly NOT changed (shortstep)

- The final `completion` segment's score is still graded on evaluation outcome (5 correct, 1 wrong, 3 near-miss), independent of step count. Step-count pressure applies to intermediate subgoals.
- JSON output schema (`start_step`, `end_step`, `subgoal`, `type`, `contribution`, `rationale`).
- The `VALID_TYPES` set, size guards (`MAX_TRAJECTORY_BYTES`, `MAX_STEPS`), retry logic, pricing table.
- Pretty-dump structure.

## Changes inside `run_rollouts_KS_giveup.py`

### Prompt

**Unchanged** — uses the baseline `JOINT_SEGMENT_REWARD_PROMPT` verbatim. The variant's effect is purely post-processing.

### New post-processor: `truncate_at_abandoned_failure(segments)`

Inserted in `process_rollout()` after `merge_consecutive_same_score()` and before the contribution-stat aggregation. Pseudocode:

```python
def truncate_at_abandoned_failure(segments: list[dict]) -> tuple[list[dict], Optional[int]]:
    """Return (kept_segments, cut_at_step_or_None).
    Truncate at the first failed (contribution<=2) segment whose successor
    has a different subgoal AND a different type. The failed segment itself
    is dropped along with everything after it. Returns the new effective
    end_step (the end_step of the last KEPT segment) so the trajectory text
    and num_steps can be re-derived consistently.
    """
    for i, seg in enumerate(segments):
        if seg["contribution"] > 2:
            continue
        if i + 1 >= len(segments):
            continue  # failed segment is last; nothing to truncate after
        nxt = segments[i + 1]
        same_subgoal = (seg.get("subgoal", "").strip().lower()
                        == nxt.get("subgoal", "").strip().lower())
        same_type = (seg.get("type") == nxt.get("type"))
        if same_subgoal or same_type:
            continue  # legitimate retry, keep
        kept = segments[:i]
        if not kept:
            return segments, None  # nothing kept — fall back to no truncation
        return kept, kept[-1]["end_step"]
    return segments, None
```

After truncation, when `cut_at_step` is not None, also:

- Filter `steps` to only those with `step <= cut_at_step`, then re-build `trajectory_text = format_trajectory(steps_kept)`.
- Recompute `num_steps`, `num_segments`, `num_raw_segments`, `mean_contribution`, `min_contribution`, `max_contribution` on the kept segments.
- Add two new top-level fields to the saved record so the truncation is auditable:
  - `"truncated_at_step": cut_at_step` (or `null` if no truncation applied)
  - `"truncated_segments_dropped": <int>` (count of segments removed, 0 if untruncated)

These are additive fields only — the existing schema fields all remain present.

The `raw_segments` field (pre-merge model output) is also preserved untruncated under the existing key so the original model output is always recoverable for debugging.

### Same-subgoal heuristic note

The pre-merge `subgoal` strings come from the model and may differ in wording even when conceptually the same. The conservative rule above ("different subgoal AND different type") errs toward **keeping** segments — a borderline case where the model uses slightly different wording for the retry will still be classified as a legitimate retry because the `type` will match. The intended trigger is the clean case: failed `data_fetch: get contacts` followed by `action: send message` (the agent moved on to a different phase entirely).

### Output paths and resume

Writes to `results/rollout/giveup_<seed>.jsonl` and `results/rollout/pretty_giveup/`. `load_already_done()` updated to read from `giveup_<seed>.jsonl`.

### What is explicitly NOT changed (giveup)

- The prompt sent to the model.
- The rubric, scoring bands, and 7-type taxonomy.
- The fact that the model sees the full untruncated trajectory at scoring time.

## Verification

This spec does not require automated tests. After implementation, sanity-check by running smoke mode on a few seed-1 tasks for each variant:

```
python rubric_reward/run_rollouts_KS_shortstep.py --seed 1 --limit 3
python rubric_reward/run_rollouts_KS_giveup.py    --seed 1 --limit 3
```

and visually compare the resulting `pretty_shortstep/seed_1__*.json` and `pretty_giveup/seed_1__*.json` against the existing baseline `pretty/seed_1__*.json` for the same tasks.

Expectations:

- **Shortstep:** on rollouts that contain a hallucinated-API crash followed by a successful recovery, the recovery segment scores 4 versus 4–5 in the baseline; clean trajectories largely unchanged.
- **Giveup:** records that include a failed-and-abandoned subgoal have `truncated_at_step` set and `num_steps` ≤ baseline. Records without that pattern have `truncated_at_step: null` and are byte-equivalent on shared fields to the baseline record.

Full 720-rollout runs and downstream SeRPO ablations are out of scope for this spec — they are training-pipeline work tracked separately.

## Out of scope

- Changing the baseline rubric or the prompt used by `_giveup`.
- Schema changes other than the two additive `_giveup` fields (`truncated_at_step`, `truncated_segments_dropped`).
- Per-step subtraction rules or counterfactual-step estimation.
- Subgoal-group clustering as a separate output structure.
- A combined `_shortstep_giveup` variant. Each effect gets its own file so they can be ablated independently; combining is left for a future spec once each variant's effect is measured.
- Pre-pass truncation (truncating the prompt input before scoring). The model must see the full trajectory to assign correct contributions.
