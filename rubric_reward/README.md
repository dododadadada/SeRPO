# Joint Rubric Scorer — Variant Reference

Four `run_rollouts_KS_*.py` scripts live in this directory. They all do the same thing — read AppWorld rollouts, call GPT-5-mini with the joint segmentation + 1–5 scoring prompt, validate and merge segments, write JSONL + pretty JSON — and differ only in **prompt wording**, **post-processing**, and **output paths**. Output paths are kept distinct so resume logic and downstream consumers can't cross-contaminate.

## Quick comparison

| Variant | Prompt change | Code (post-processing) | Schema | Output directory |  
|---|---|---|---|---|
| `_baseline` | original 7-type, original rubric | none | original | `results/rollout/<round>/KS_baseline/` |
| `_shortstep` | +task-level cap, +efficiency anchor, +anti-split | none | original | `results/rollout/<round>/KS_shortstep/` |
| `_giveup` | identical to baseline | +`truncate_at_abandoned_failure` | +2 audit fields | `results/rollout/<round>/KS_giveup/` |
| `_noerror` | 6-type (no `error_recovery`) | `VALID_TYPES` minus `error_recovery` | original | `results/rollout/<round>/KS_noerror/` |

### Results layout by rollout generation

Outputs are grouped by which rollout generation they scored. The `<round>` directory mirrors the source folder under `appworld/experiments/outputs/rollout/`:

- `results/rollout/round0/` — scores of the **Qwen2.5-7B** rollouts (`appworld/.../rollout/round0/`)
- `results/rollout/round0_9b/` — scores of the **Qwen3.5-9B** rollouts (`appworld/.../rollout/round0_9b/`)

The four scripts currently read from `round0_9b` (set by `ROLLOUT_ROOT`) and write to `results/rollout/round0_9b/KS_<variant>/` (set by `ROUND_DIR` / `VARIANT_DIR`). To re-target a different generation, change `ROLLOUT_ROOT` and `ROUND_DIR` together — keep them in sync.

Each variant directory contains `seed_<N>.jsonl` (one record per scored rollout) plus a `pretty/` subdirectory holding human-readable per-record JSON when `--limit` is used. `failed.jsonl` and `skipped.jsonl` live at the per-generation root (`results/rollout/<round>/`).

## Shared skeleton

All four scripts:

- Read rollouts from `appworld/experiments/outputs/rollout/round0/seed_<N>/tasks/<task_id>/` (trajectory in `logs/environment_io.md`, evaluation in `evaluation/report.md`).
- Pull instructions from `rubric_reward_poc(mid_report)/data/tasks/<task_id>/instruction.txt` (shared across seeds).
- Build a single user-message prompt with the three slots `{instruction}`, `{trajectory}`, `{evaluation_summary}`.
- Call GPT-5-mini (default `reasoning_effort=medium`, `max_completion_tokens=8192`) with up to 3 retries on rate-limit / API / JSON-decode errors.
- Validate emitted segments against `VALID_TYPES`, then run `merge_consecutive_same_score` (which collapses adjacent segments sharing both `type` and `contribution`).
- Write one JSONL record per successful rollout, plus a pretty per-record JSON when `--limit` is set.
- Skip trajectories where `environment_io.md > 200 KB` or step count `> 50` (infinite-loop guard).

## 1. `_baseline.py` — the reference

The unmodified joint scorer.

- **Seven phase types:** `login`, `api_exploration`, `data_fetch`, `computation`, `action`, `completion`, `error_recovery`.
- **1–5 scoring** with the original anchors (1 = dead-end, 2 = mostly wasted, 3 = progress with overhead, 4 = productive with minor inefficiency, 5 = critical or maximally efficient).
- "Successful recovery scores 4–5 on its own merits" rule — each segment is scored independently.
- No truncation, no efficiency cap.

**Output:** `results/rollout/<round>/KS_baseline/seed_<N>.jsonl` + `.../KS_baseline/pretty/`.

**Use it as:** the control for any rubric or post-processing ablation.

## 2. `_shortstep.py` — task-level cleanliness cap

Same code as baseline. **Four prompt edits**, all in the rubric half of the prompt:

- **Task-level cap.** If any segment in the trajectory scores 1 or 2, every non-completion segment in that trajectory is bounded above by 4 — even individually clean ones on unrelated subgoals. The final `completion` segment is exempt and still scored on evaluation outcome.
- **Efficiency anchor.** "Within the same score, a trajectory that reaches its goal with fewer wasted steps overall is better than one with more wasted steps." Framed as a cross-trajectory tiebreaker, bundled with "do NOT split a coherent phase into smaller pieces just to claim each was done in few steps".
- **Anti-split guardrail** in the Granularity block: "Score-driven splitting is not allowed. Boundaries are determined by phase changes only, never by trying to maximize how many segments can claim a high score."
- **AppWorld failure-pattern example** updated so its score numbers reflect the cap (recovery segment is named as capped at 4, with the explicit note that other individually-clean segments in the same trajectory are also capped).

Schema and code identical to baseline. Net effect: fully clean trajectories still get all-5s and a 5.0 mean; trajectories with any failure get dragged down because every clean non-completion segment is now capped at 4.

| Rollout | Baseline mean | Shortstep mean |
|---|---|---|
| Fully clean (5+5+5+5) | 5.00 | 5.00 |
| Hallucinated-API + clean recovery (1+5+5) | 3.67 | **3.00** (1+4+4) |
| One failure in a long clean trajectory (1+5+5+5+5+5, last is completion) | 4.33 | **3.67** (1+4+4+4+4+5) |

**Output:** `results/rollout/<round>/KS_shortstep/seed_<N>.jsonl` + `.../KS_shortstep/pretty/`.

**Question it answers:** does penalizing the whole trajectory for any failure produce a useful GRPO advantage signal?

## 3. `_giveup.py` — abandoned-subgoal truncation

**Prompt is identical to baseline** — same 7 types, same 1–5 anchors, no cap. The difference is a new Python post-processor: `truncate_at_abandoned_failure()`.

After the model scores the full untruncated trajectory:

1. Walk the merged segments.
2. Find the first segment with `contribution ≤ 2` whose successor has both a different `subgoal` AND a different `type` (= the agent gave up on this subgoal and moved to unrelated work).
3. **Keep** that failed segment (negative training signal). **Drop** everything strictly after it.
4. Re-derive `trajectory_text` and `num_steps` from the kept prefix. Recompute contribution stats (`mean`, `min`, `max`, segment counts) on the kept segments.

The conservative "different subgoal AND different type" rule errs toward keeping segments — a same-`type` retry with slightly different wording is treated as a legitimate retry, not an abandonment.

Two **additive** audit fields are added to each saved record:

- `truncated_at_step` — `end_step` of the failed-and-kept segment, or `null` if no truncation applied.
- `truncated_segments_dropped` — count of segments dropped from after the failed one, `0` if untruncated.

Records with no abandoned-failure pattern are equivalent to baseline on all shared fields.

**Output:** `results/rollout/<round>/KS_giveup/seed_<N>.jsonl` + `.../KS_giveup/pretty/`.

**Question it answers:** does stripping post-abandonment steps de-noise the per-rollout reward without distorting the model's scoring?

## 4. `_noerror.py` — `error_recovery` removed from the phase taxonomy

Two coordinated changes against baseline:

- **Prompt:** the `error_recovery` bullet is removed from the phase-type list. Header changes from "seven phase types" to "six phase types".
- **Code:** `VALID_TYPES` drops `error_recovery`. Any segment the model emits with `type: "error_recovery"` is dropped by `validate_segments` rather than scored.

Practically, this forces the model to classify recovery work by what it *is doing* (`api_exploration`, `data_fetch`, etc.) rather than by *why* it's there. The "this segment is a recovery" information now has to live in `subgoal` or `rationale` instead of `type`. The 1–5 anchors and the "successful recovery scores 4–5" rule are unchanged from baseline, so scoring still rewards recoveries — they just aren't named as such.

**Output:** `results/rollout/<round>/KS_noerror/seed_<N>.jsonl` + `.../KS_noerror/pretty/`.

**Question it answers:** does the `error_recovery` label add useful signal, or does it create a redundant category that just confuses the segmenter?

## How the four relate as an ablation

- **baseline** is the control.
- **shortstep** changes the *rubric* (how scores are assigned).
- **giveup** changes the *trajectory the trainer sees* (post-processing the saved record).
- **noerror** changes the *taxonomy* (what phase labels the model can use).

The three non-baseline effects are pairwise orthogonal: any combination could be stacked by editing two scripts together. Keeping each in its own file means each effect can be measured cleanly without confounding.

## Usage

All four scripts share the same CLI:

```
python rubric_reward/run_rollouts_KS_<variant>.py                          # default round (round0_9b), all 8 seeds
python rubric_reward/run_rollouts_KS_<variant>.py --round round1_9b        # score a different rollout generation
python rubric_reward/run_rollouts_KS_<variant>.py --seed 1                 # one seed only
python rubric_reward/run_rollouts_KS_<variant>.py --limit 5                # 5 per seed (smoke test, writes pretty JSON)
python rubric_reward/run_rollouts_KS_<variant>.py --dry-run                # cost estimate, no API calls
python rubric_reward/run_rollouts_KS_<variant>.py --workers 16             # higher parallelism
python rubric_reward/run_rollouts_KS_<variant>.py --model gpt-5            # different model
```

Flags compose, so a typical batch run looks like:

```bash
for v in baseline shortstep giveup noerror; do
  for s in 1 2 3 4 5 6; do
    python rubric_reward/run_rollouts_KS_${v}.py --round round1_9b --seed $s
  done
done
```

`--round <name>` switches both the rollout source (`appworld/.../rollout/<name>/`) and the results subtree (`results/rollout/<name>/KS_<variant>/`) in one step. The script exits early if the named source directory doesn't exist, so a typo is caught immediately.

Resume is automatic — each script reads its own `<round>/KS_<variant>/seed_<N>.jsonl` and skips task IDs already present, so re-running is safe. Note: resume keys on task ID only, so a variant directory must hold scores for exactly one rollout generation — never point two generations at the same `<round>/KS_<variant>/` folder.
