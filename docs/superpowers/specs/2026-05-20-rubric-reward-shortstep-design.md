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

### `_shortstep`: task-level cleanliness cap

Inject a **task-level cleanliness cap** into the existing 1–5 rubric so that:

- A clean trajectory (no segment with `contribution <= 2`) can earn segments scoring up to 5.
- If **any** segment in the trajectory failed (1 or 2), **every other non-completion segment** in that trajectory is capped at 4, regardless of whether they were individually clean or pursued unrelated subgoals.
- The final `completion` segment is exempt from the cap — it is still graded on evaluation outcome only (5 correct, 1 wrong, 3 near-miss).
- A failed segment continues to score 1–2 as in the baseline.

**Why task-level, not per-subgoal.** An earlier per-subgoal cap (cap only segments pursuing the same subgoal as an earlier failure) had a known failure mode: it incentivized the model to over-split a coherent phase into many small "subgoals" so each tiny piece could claim "clean first attempt = 5". The task-level rule removes that incentive entirely — splitting cannot manufacture more 5s, because the existence of any failed segment caps the whole trajectory.

The cap is preferred over a per-step subtraction rule because the LLM can reliably detect "did anything fail in this trajectory?" but is unreliable at estimating a counterfactual minimum step count for AppWorld tasks.

**Anti-over-splitting guardrails.** Two prompt-level guardrails reinforce the task-level cap:

1. The Efficiency anchor is reframed as a *cross-trajectory* tiebreaker (fewer wasted steps overall), with an explicit prohibition: "do NOT split a coherent phase into smaller pieces just to claim each piece was 'done in few steps'."
2. The Granularity block adds: "Score-driven splitting is not allowed. Boundaries are determined by phase changes only, never by trying to maximize how many segments can claim a high score."

### `_giveup`: post-processing truncation after abandoned subgoals

Keep the baseline rubric prompt unchanged. After the model returns segments for the full trajectory, walk the segments and **truncate everything after the first abandoned-subgoal failure**:

- A segment is "failed" if its `contribution` ≤ 2.
- A failed segment is "abandoned" if the next segment has a different `subgoal` AND different `type` (i.e., the agent moved on rather than retrying).
- When this pattern is found, **keep the failed segment** and drop every segment strictly after it. The failed segment is retained because it is the negative training signal the policy needs to learn from; the discarded segments are the ones that pollute the per-rollout reward because they worked on a different subgoal.
- Re-derive `trajectory_text` and `num_steps` from the kept range so downstream consumers see a consistent record.
- A failed segment followed by a same-subgoal recovery attempt is left intact — that's a legitimate retry and is what `_shortstep` is designed to score.
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

### Prompt edits (four edits to `JOINT_SEGMENT_REWARD_PROMPT`)

Everything else — template slots `{instruction}`, `{trajectory}`, `{evaluation_summary}`; the 7 phase types; the 1–5 contribution schema; the output JSON shape; the post-processing pipeline (`validate_segments`, `merge_consecutive_same_score`) — is unchanged.

**Edit 1 — new "Efficiency" anchor inside the 1–5 rubric block (reframed as cross-trajectory tiebreaker with anti-split prohibition):**

> **Efficiency matters within a score band.** Within the same score, a trajectory that reaches its goal with fewer wasted steps overall is better than one with more wasted steps. Use this as a tiebreaker between similar segments; do NOT split a coherent phase into smaller pieces just to claim each piece was "done in few steps". Segmentation granularity is governed by the segmentation rules above, not by score-maximization.

**Edit 2 — task-level cleanliness cap in "Anchoring principles":**

> **Score 5 requires a fully clean trajectory (task-level cap).** Score 5 is reserved for segments in trajectories where **no segment** has `contribution <= 2`. If **any** segment in this trajectory failed (1 or 2) — regardless of whether it was the same subgoal or a different one — then no other segment in the trajectory can score above 4. This is a per-trajectory rule, not per-subgoal: one failed segment anywhere caps every non-completion segment at 4. Rationale: the policy we want is one that completes the whole task cleanly; partial cleanliness on independent subgoals does not earn the top score. The final `completion` segment is the only exception.

**Edit 3 — update the existing AppWorld failure-pattern block to reflect task-level scope:**

> The subsequent recovery segment that finds the right API and makes progress gets **4** (capped at 4 because this trajectory contains a failed segment) — it did real productive work, even if it wouldn't have been needed without the earlier mistake.
>
> All other non-completion segments in this trajectory (e.g., the earlier clean `login` segment, an unrelated clean `data_fetch` later on) are also capped at 4 by the task-level rule above, even though they were individually clean. Only the final `completion` segment can still score 5 if the answer is correct.

**Edit 4 — anti-over-splitting line in the Granularity block:**

> **Score-driven splitting is not allowed.** Boundaries are determined by phase changes only, never by trying to maximize how many segments can claim a high score. A coherent multi-step phase stays as ONE segment even if splitting it would let smaller pieces look "cleaner". If you find yourself considering a boundary because it would change a contribution value, ignore that consideration — the segmentation rules above are the only valid reason to draw a boundary.

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
    Find the first failed (contribution<=2) segment whose successor has a
    different subgoal AND a different type (= subgoal was abandoned). KEEP
    that failed segment (it is the negative training signal) and drop
    everything strictly AFTER it. Returns the new effective end_step (the
    failed segment's end_step) so the trajectory text and num_steps can be
    re-derived consistently.
    """
    for i, seg in enumerate(segments):
        if seg["contribution"] > 2:
            continue
        if i + 1 >= len(segments):
            continue  # failed segment is last; nothing to truncate
        nxt = segments[i + 1]
        same_subgoal = (seg.get("subgoal", "").strip().lower()
                        == nxt.get("subgoal", "").strip().lower())
        same_type = (seg.get("type") == nxt.get("type"))
        if same_subgoal or same_type:
            continue  # legitimate retry, keep everything after
        kept = segments[:i + 1]  # include the failed segment itself
        return kept, kept[-1]["end_step"]
    return segments, None
```

After truncation, when `cut_at_step` is not None, also:

- Filter `steps` to only those with `step <= cut_at_step`, then re-build `trajectory_text = format_trajectory(steps_kept)`. This keeps the steps that belong to the failed-and-kept segment and drops everything after.
- Recompute `num_steps`, `num_segments`, `num_raw_segments`, `mean_contribution`, `min_contribution`, `max_contribution` on the kept segments.
- Add two new top-level fields to the saved record so the truncation is auditable:
  - `"truncated_at_step": cut_at_step` (= end_step of the failed segment, or `null` if no truncation applied)
  - `"truncated_segments_dropped": <int>` (count of segments removed from after the failed one, 0 if untruncated)

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

- **Shortstep:** on rollouts that contain any failed segment (1 or 2), **all** non-completion segments are capped at 4 — including segments that were individually clean and pursued unrelated subgoals. Fully clean trajectories are unchanged from baseline. Segment counts should be similar to baseline (no over-splitting); if a shortstep rollout produces noticeably more segments than its baseline counterpart, the anti-split guardrails are not landing and the prompt needs revisiting.
- **Giveup:** records that include a failed-and-abandoned subgoal have `truncated_at_step` set and `num_steps` ≤ baseline. Records without that pattern have `truncated_at_step: null` and are byte-equivalent on shared fields to the baseline record.

Full 720-rollout runs and downstream SeRPO ablations are out of scope for this spec — they are training-pipeline work tracked separately.

## Out of scope

- Changing the baseline rubric or the prompt used by `_giveup`.
- Schema changes other than the two additive `_giveup` fields (`truncated_at_step`, `truncated_segments_dropped`).
- Per-step subtraction rules or counterfactual-step estimation.
- Subgoal-group clustering as a separate output structure.
- A combined `_shortstep_giveup` variant. Each effect gets its own file so they can be ablated independently; combining is left for a future spec once each variant's effect is measured.
- Pre-pass truncation (truncating the prompt input before scoring). The model must see the full trajectory to assign correct contributions.
