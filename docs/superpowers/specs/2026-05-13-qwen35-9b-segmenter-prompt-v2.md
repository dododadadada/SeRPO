# Qwen3.5-9B Segmenter Prompt v2 (Coarse Granularity)

**Date:** 2026-05-13
**Author:** brainstormed with Claude
**Goal:** Improve subgoal-label quality and reduce incorrect merges in the Qwen3.5-9B (reasoning OFF) segmenter, **while preserving its naturally coarse granularity**. Do NOT optimize against Claude's finer-grained gold.

## Context

90 AppWorld tasks segmented with `qwen/qwen3.5-9b` (reasoning OFF, prompt v1, OpenRouter):

- F1 vs Claude gold: exact 0.807 / ±1-tol 0.850
- Precision 0.921 / Recall 0.742 — systematic **under-segmentation**
- Mean −1.37 segments per task vs Claude

User decision (this brainstorming): **coarse granularity is the right target**. The recall gap is not a defect — Claude's "split on every phase_type change" is a different (also valid) choice.

Failure modes worth fixing inside the coarse regime:

1. Subgoal labels are generic ("Identify and retrieve roommate contact information" for 10 heterogeneous steps).
2. Occasionally merges two genuinely different sub-tasks.
3. segment_type sometimes describes the latest step only, not the segment's actual goal.

## Design

Add a v2 prompt in `rubric_reward_poc/segment_trajectories.py` next to `LLM_SEGMENT_PROMPT` (call it `LLM_SEGMENT_PROMPT_V2`). Make `segment_qwen_openrouter.py` select v1 or v2 by CLI flag `--prompt-version v1|v2`.

Key changes from v1:

- Replace "natural transition point (e.g., switching apps, starting a new phase, recovering from an error)" with an **explicit closed list** of boundary conditions:
  - (a) the agent switches to a *different* app to start a different sub-task,
  - (b) the previous sub-task completes (or fails permanently) and a distinct new one begins,
  - (c) explicit error-recovery attempt.
- Explicitly forbid splitting on intra-sub-task phase changes: *"Do NOT split when the agent moves between api_exploration, login, and data_fetch within one sub-task — that is one segment."*
- Strengthen the subgoal label requirement: *"subgoal must describe the sub-task's end goal (what does this segment accomplish?), not the most recent step."*
- segment_type rule: *"Describe the dominant goal of the whole segment, even if some constituent steps would individually be labeled differently."*

Out of scope:

- GPU / local vLLM (GPU full)
- LLM-judge evaluation (Claude-as-judge would be the standard tool; user prefers manual qualitative analysis by the assistant)
- F1-vs-Claude as the primary metric

## Execution

1. Add `LLM_SEGMENT_PROMPT_V2` to `segment_trajectories.py`.
2. Add `--prompt-version` flag to `segment_qwen_openrouter.py`.
3. Run on all 90 tasks: `python segment_qwen_openrouter.py --variant a --prompt-version v2 --label qwen35_9b_a_v2`.
4. Assistant compares v1 vs v2 on ~5–10 highest-disagreement tasks: for each, present the trajectory-relevant boundaries, both segmentations side-by-side, and write a short qualitative judgment (improved / regressed / neutral, and why).
5. Summarize: did v2 reduce generic-label cases? did it introduce new over-merges?

## Success criteria

Qualitative only:

- v2 produces more specific, sub-task-grounded subgoal labels.
- Fewer "Identify and retrieve X" style omnibus merges.
- No new regressions where genuinely different sub-tasks are merged.
- Segment counts roughly comparable to v1 (this is *not* a granularity change).

## Notes

- Repo is not a git repository (per environment), so no commit step.
- v1 output preserved at `results/segments_qwen35_9b_a.jsonl`; v2 saves to `results/segments_qwen35_9b_a_v2.jsonl` (the existing `--label` flag handles this).
- vLLM serve script (`serve_qwen35_9b.sh`) already drafted; revisit when GPU frees up to swap from OpenRouter to local.

---

## Revision: Path B (Fine Granularity) — 2026-05-13

**Status:** v2 ran and is preserved as a reference, but **the design above is superseded.** Direction reversed after observing an internal contradiction.

### What went wrong with v2

User flagged that the `segment_type` taxonomy `[login, api_exploration, data_fetch, computation, action, completion, error_recovery]` was kept unchanged in v2, while the new rules said "do NOT split on login → api_exploration → data_fetch within one sub-task". These conflict: if `login` is a sub-task *type*, then a login *is* its own segment; if login is a phase to merge, it shouldn't be in the type list.

Empirical check on Claude's 90-task gold (618 segments):

| type | freq | avg steps/seg | 1-step% |
|---|---|---|---|
| data_fetch | 23.6% | 1.75 | 50% |
| **login** | **18.0%** | 2.07 | 51% |
| **api_exploration** | **17.2%** | 2.23 | 14% |
| completion | 14.2% | 1.12 | 88% |
| action | 11.8% | 1.42 | 70% |
| error_recovery | 8.4% | 2.62 | 25% |
| computation | 6.8% | 1.57 | 62% |

login + api_exploration = 35% of all Claude segments. The taxonomy is built for fine granularity — Claude treats both as legitimate stand-alone sub-tasks.

v2 measured: avg 4.41 segs/task (vs Claude 6.87, v1 5.50). v2 went *further* from Claude than v1 — wrong direction relative to what the taxonomy implies.

### Revised goal (v3, Path B)

Target Claude-style fine granularity. F1 vs Claude becomes the right metric again. Target: 0.81 → 0.90+.

### v3 prompt strategy

Stay with Option A (explicit rules; no few-shot). Key changes vs v1:

- Define each of the 7 phase types with concrete in-domain examples (e.g., `login` = "calling spotify.login(...)").
- Explicit "phase change = new segment" rule, with concrete cases (app switch → segment; api_exploration → login within same app → segment; data_fetch → computation → segment).
- Granularity hint: "AppWorld trajectories typically have 6–12 segments. Most segments are 1–3 steps. A single login() call is its own segment."

### v3 execution

1. Add `LLM_SEGMENT_PROMPT_V3` next to v1/v2 in `segment_trajectories.py`.
2. Register in `PROMPT_VERSIONS` dict; `--prompt-version v3` already works via existing CLI plumbing.
3. Run: `python segment_qwen_openrouter.py --variant a --prompt-version v3 --label qwen35_9b_a_v3`.
4. Extend `compare_segmenters.py` to load v2 and v3 alongside the existing A (v1).
5. Assistant qualitative analysis on top-disagreement tasks (v1 vs v3).

### v3 success criteria

- F1 vs Claude > 0.85 (exact) and > 0.90 (±1-tolerance)
- Average segments/task close to Claude's 6.87 (range 6–8)
- login + api_exploration segment share comparable to Claude's 35%
- No new failure modes (every-step segmentation, missing completion, etc.)

---

## Architecture Pivot: Joint Segmentation + Per-Segment Reward (2026-05-13)

**Status:** v3 90-task run was canceled before launching. The taxonomy-debugging conversation expanded into a deeper architectural question: should segmentation and reward be one call or two?

### Empirical justification for joint

Failure-rate analysis on the 90 trajectories:

- 72% of trajectories contain ≥1 `Execution failed` step.
- 100% of tasks ultimately succeed (per evaluation.md) — so task-level outcome reward is constant (+1) and gives no GRPO gradient signal.
- All meaningful learning signal must come from **per-segment differentiation**.
- Per-task root-cause attribution on the 65 failed trajectories: **78.5% of failures originate from inadequate api_exploration** (agent calls non-existent APIs like `get_contacts`, `list_notes`, `list_files`). The cause segment is often *absent*, not just bad — the agent skipped exploration.

→ This pattern argues strongly for joint reasoning: the model needs to see segment boundaries AND assess "should this action have been preceded by exploration?" simultaneously. Splitting into two passes duplicates the trajectory context and loses the link between "where the boundary is" and "why this segment was wasteful."

### Design decisions

1. **Drop trajectory-level rubric from the joint output.** Reasons:
   - GRPO broadcasts per-segment reward to tokens. Trajectory reward applied uniformly = zero advantage. Not training-useful.
   - Existing trajectory rubric in `rubric_reward_poc/results/rubric_scores.jsonl` (90 tasks already scored) is preserved as legacy/analysis.
2. **Per-segment scalar 1–5 contribution score.**
   - Scale chosen to match existing trajectory rubric for calibration reuse and consistent UX.
   - GRPO group-relative normalization `(r − μ)/σ` makes the absolute scale irrelevant; "1" becomes a large negative advantage automatically when matched segments score higher in other rollouts.
3. **New directory:** `/data/minjeong/Autonomous_agent/rubric_reward/` (separate from `rubric_reward_poc/` so the POC artifacts stay untouched).
4. **Initial model:** GPT-5-mini via OpenRouter (`openai/gpt-5-mini`). Smoke test before committing to 90 tasks. Cost estimate ≈ $0.70 for full 90.
5. **Phase taxonomy:** keep the 7 types (login, api_exploration, data_fetch, computation, action, completion, error_recovery) — they are descriptive labels for human inspection, and consistent with Claude gold for cross-comparison. Type label is not the load-bearing signal for SeRPO; the `contribution` scalar is.

### Output format

```json
{
  "task_id": "...",
  "instruction": "...",
  "num_steps": 22,
  "model": "openai/gpt-5-mini",
  "segments": [
    {"start_step": 1, "end_step": 1, "subgoal": "Login to phone", "type": "login", "contribution": 4, "rationale": "..."},
    ...
  ],
  "num_segments": 9,
  "mean_contribution": 3.4
}
```

### Files

- `rubric_reward/joint_score.py` — prompt + loaders + run loop, self-contained
- `rubric_reward/results/joint_<model_label>.jsonl` — output

### Success criteria (joint smoke)

- Output parses on 5 sample tasks without retries.
- Contribution scores have within-task spread (not all 3s).
- "Premature action" failures (api_exploration deficit) are scored 1–2 in the failing segment.
- "Recovery after error" segments are scored 2–3.
- Final successful completion segment is 4–5.
- Boundaries on the smoke set look sensible to manual inspection.
