"""
run_rollouts_KS_shortstep.py — Joint segmentation + per-segment reward scoring
with a SHORT-STEP preference baked into the rubric.

Variant of run_rollouts_KS_baseline.py. Differences:
- Prompt rubric adds an efficiency anchor and caps successful-recovery
  segments at contribution 4 (clean first-attempt = 5).
- Writes to results/rollout/round0_9b/KS_shortstep/seed_<N>.jsonl and
  round0_9b/KS_shortstep/pretty/ so it doesn't collide with the baseline's
  round0_9b/KS_baseline/seed_<N>.jsonl.

Reads rollouts from
  appworld/experiments/outputs/rollout/round0_9b/seed_<N>/tasks/<task_id>/
with trajectory in logs/environment_io.md and evaluation in evaluation/report.md.

Instructions are pulled from rubric_reward_poc(mid_report)/data/tasks/<task_id>/instruction.txt
(instructions are shared across seeds).

Skips trajectories where environment_io.md > 200KB OR step count > 50 (infinite-loop guard).
Failed scorings (after 3 retries) are recorded in failed.jsonl; skipped rollouts in skipped.jsonl.

Parallelism: ThreadPoolExecutor (default 8 workers).

Usage:
  python run_rollouts.py                        # all 720
  python run_rollouts.py --seed 1               # one seed only
  python run_rollouts.py --limit 5              # 5 per seed (smoke)
  python run_rollouts.py --dry-run              # cost estimate, no API calls
  python run_rollouts.py --workers 16           # higher parallelism
  python run_rollouts.py --model gpt-5          # different model
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

# =============================================================================
# Paths & config
# =============================================================================

REPO = Path("/data/minjeong/Autonomous_agent")
BASE_TASKS_DIR = REPO / "rubric_reward_poc(mid_report)" / "data" / "tasks"
THIS_DIR = REPO / "rubric_reward"
RESULTS_DIR = THIS_DIR / "results" / "rollout"

# Rollout source root (one subdirectory per rollout generation).
ROLLOUTS_ROOT = REPO / "appworld" / "experiments" / "outputs" / "rollout"

# Default rollout generation, used if --round is not given.
DEFAULT_ROUND = "round0_9b"

# Variant-specific results subdirectory name (each script sets this differently).
VARIANT_NAME = "KS_shortstep"

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_WORKERS = 8
DEFAULT_REASONING = "medium"
MAX_RETRIES = 3
MAX_COMPLETION_TOKENS = 8192
MAX_TRAJECTORY_BYTES = 200_000  # 200KB — covers 990MB outlier and similar runaways
MAX_STEPS = 50                   # infinite-loop guard
RETRY_BASE_WAIT = 2.0            # exponential backoff base
RATE_LIMIT_WAIT = 30.0           # initial wait for 429
VALID_TYPES = {
    "login", "api_exploration", "data_fetch", "computation",
    "action", "completion", "error_recovery",
}

# OpenAI pricing per 1M tokens (May 2026 rates; update if pricing changes)
MODEL_PRICING = {
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5":      {"input": 1.25, "output": 10.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
}

load_dotenv(REPO / ".env")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not found in environment.", file=sys.stderr)
    print("Add to /data/minjeong/Autonomous_agent/.env: OPENAI_API_KEY=sk-...", file=sys.stderr)
    sys.exit(1)


# =============================================================================
# Joint prompt (verbatim from validated POC; merge rule kept inline)
# =============================================================================

JOINT_SEGMENT_REWARD_PROMPT = """\
You are analyzing an AI agent trajectory on an AppWorld tool-use task. You have ONE job split in two parts that must be done together:

1. **Segment** the trajectory into phases (each phase = one or more consecutive steps sharing a single subgoal).
2. For each segment, assign a **contribution score (1-5)** reflecting how productively that segment advanced the task, given the trajectory as a whole.

## Task
{instruction}

## Trajectory
{trajectory}

## Evaluation result
{evaluation_summary}

---

## Segmentation rules

A "phase" is one coherent activity. There are seven phase types:

- **login**: authenticating with an app (e.g., `apis.spotify.login(...)`, often including retries to obtain a valid token)
- **api_exploration**: looking up which APIs exist or reading their specs (e.g., `apis.api_docs.show_api_descriptions(...)`, `apis.api_docs.show_api_doc(...)`)
- **data_fetch**: retrieving information from an app or filesystem (e.g., listing artists, reading a file, looking up a contact); may include pagination
- **computation**: local processing of already-fetched data (e.g., filtering, aggregating, formatting)
- **action**: performing a goal-relevant action that changes external state (e.g., sending a message, making a payment, following artists)
- **completion**: marking the task complete (e.g., `apis.supervisor.complete_task(...)`)
- **error_recovery**: after a failure (`Execution failed` / `Traceback`), the agent explicitly changes approach to recover

### Start a new segment when
- The agent switches apps to start a different sub-task.
- The phase type changes within the same app (api_exploration → login, login → data_fetch, data_fetch → computation, computation → action).
- A distinct new action begins (different target, different intent).
- The agent recovers from an error by changing approach.

### Keep steps in the same segment when
- Consecutive steps of the SAME phase type working toward the SAME intermediate subgoal. Example: `show_api_descriptions(...)` then `show_api_doc(api_name=...)` are both api_exploration toward "understand this app's API" → ONE segment.
- The agent retries the same action with corrected inputs (still the same subgoal).
- A loop of repeated calls accomplishing one goal (e.g., `follow_artist(id_1)`, `follow_artist(id_2)`, ...) → ONE action segment.

### Granularity
AppWorld trajectories typically have **6–12 segments total**, NOT one segment per step. Most segments span 1–3 steps. A single 1-step segment is fine when it IS a complete phase on its own (e.g., a lone `login(...)` or `apis.supervisor.complete_task(...)`).

**Score-driven splitting is not allowed.** Boundaries are determined by phase changes only, never by trying to maximize how many segments can claim a high score. A coherent multi-step phase stays as ONE segment even if splitting it would let smaller pieces look "cleaner". If you find yourself considering a boundary because it would change a contribution value, ignore that consideration — the segmentation rules above are the only valid reason to draw a boundary.

### Avoid redundant boundaries
If two consecutive segments would end up with the SAME `type` AND the SAME `contribution`, merge them into ONE segment. A boundary that doesn't change either of those values is not adding signal. Difference in `type` or in `contribution` between adjacent segments is the JUSTIFICATION for the boundary.

### Non-overlapping segments
Segments must partition the trajectory: `segment[i+1].start_step > segment[i].end_step`. Each step belongs to exactly ONE segment. If one step crams multiple phases into a single script, emit ONE segment and pick the dominant phase (`action` or `completion` outranks setup phases like `login` or `data_fetch`).

---

## Contribution scoring (1-5)

How much did this segment productively advance the task, considering everything before and after?

- **5**: Critical or maximally efficient. Made decisive progress on the right thing, OR executed cleanly with no waste. The final successful `completion` segment is typically 5.
- **4**: Productive with mostly correct choices. Minor inefficiency or one small recoverable mistake within the segment.
- **3**: Made progress, but with significant overhead — recovery from a mistake, partial pagination miss, redundant exploration that turned out unnecessary, or a segment that succeeded only because it was retrying after an earlier mistake.
- **2**: Mostly wasted effort. Wrong API name, wrong parameters, attempted something that had to be redone. The segment did something but most of it was lost work.
- **1**: Pure dead-end or failure. Premature action that crashed (e.g., calling a non-existent API like `get_contacts`), totally wrong approach, hallucinated API. The segment's main contribution was to teach the agent it was wrong.

**Efficiency matters within a score band.** Within the same score, a trajectory that reaches its goal with fewer wasted steps overall is better than one with more wasted steps. Use this as a tiebreaker between similar segments; do NOT split a coherent phase into smaller pieces just to claim each piece was "done in few steps". Segmentation granularity is governed by the segmentation rules above, not by score-maximization.

### Anchoring principles
- **Score each segment on its own outcome**, not on whether earlier segments were clean. A successful recovery is genuinely valuable and earns a high score.
- A "wasted" segment can still get 2-3 if it produced information that was directly used in the next segment (i.e., the error message led to recovery).
- A *failed* attempt — regardless of whether it's the first try or a recovery attempt — scores 1-2. If the agent tries to recover and the recovery itself also fails, that recovery segment is 1-2.
- A *successful* recovery — agent failed earlier, then found the right approach and made progress — scores on its own merits (typically 4 if the recovery was clean and effective; see cap below).
- **Score 5 requires a fully clean trajectory (task-level cap). This is a per-trajectory rule, not per-subgoal: one failed segment anywhere caps every non-completion segment at 4. Rationale: the policy we want is one that completes the whole task cleanly; partial cleanliness on independent subgoals does not earn the top score. The final `completion` segment is the only exception.
- The PRIOR segment that caused the need for recovery stays at 1-2. We do not retroactively penalize successful recoveries beyond the cap; the mistake itself is already penalized in its own segment.
- The final `completion` segment: 5 if the answer is correct (per evaluation result), 1 if wrong, 3 if a near-miss (mostly-correct answer or partial pass). The recovery cap does NOT apply to the final completion segment — it is graded on outcome only.
- A `complete_task(status="fail")` call — or any other "I give up" completion that does not deliver the task goal — is outcome score **1**, regardless of whether the API call itself executed successfully. Score the completion segment by whether the task goal was achieved, not by whether the completion API was invoked without error. The agent submitting "fail" is the agent reporting its own failure; do not reward graceful surrender.

### Common failure pattern to attribute correctly
The most common AppWorld failure: agent hallucinates an API name (`get_contacts`, `list_notes`, `list_files`) and crashes.
- The CRASHING segment gets contribution **1** (premature action, no exploration).
- The subsequent recovery segment that finds the right API and makes progress gets **4** (capped at 4 because this trajectory contains a failed segment) — it did real productive work, even if it wouldn't have been needed without the earlier mistake.
- All other non-completion segments in this trajectory (e.g., the earlier clean `login` segment, an unrelated clean `data_fetch` later on) are also capped at 4 by the task-level rule above, even though they were individually clean. Only the final `completion` segment can still score 5 if the answer is correct.
- If the recovery ALSO fails (e.g., agent guesses another wrong API after the first crash), that recovery segment is **1-2**.

---

## Output format

Respond ONLY with a JSON array. No markdown fences, no extra prose.

[
  {{"start_step": <int>, "end_step": <int>, "subgoal": "<3–8 words>", "type": "<one of the 7 types>", "contribution": <1-5>, "rationale": "<one short sentence, <25 words>"}},
  ...
]
"""


# =============================================================================
# Rollout data parsers
# =============================================================================

SECTION_RE = re.compile(r"###\s*Environment Interaction\s*(\d+)", re.IGNORECASE)
CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)


def parse_environment_io(text: str) -> list[dict]:
    """Parse environment_io.md into [{step, agent, env}, ...]."""
    sections = SECTION_RE.split(text)
    # sections layout: [preamble, "1", body1, "2", body2, ...]
    steps: list[dict] = []
    for i in range(1, len(sections), 2):
        try:
            step_num = int(sections[i])
        except ValueError:
            continue
        body = sections[i + 1] if i + 1 < len(sections) else ""
        blocks = CODE_BLOCK_RE.findall(body)
        agent_code = blocks[0][1].strip() if blocks else ""
        env_output = blocks[1][1].strip() if len(blocks) >= 2 else ""
        steps.append({"step": step_num, "agent": agent_code, "env": env_output})
    return steps


def format_trajectory(steps: list[dict]) -> str:
    lines: list[str] = []
    for s in steps:
        lines.append(f"[Step {s['step']} - AGENT]\n{s['agent']}")
        if s["env"]:
            lines.append(f"[Step {s['step']} - ENV]\n{s['env']}")
    return "\n\n".join(lines)


def load_instruction(task_id: str) -> Optional[str]:
    p = BASE_TASKS_DIR / task_id / "instruction.txt"
    if not p.exists():
        return None
    return p.read_text().strip()


def load_evaluation(report_path: Path) -> str:
    """Compact evaluation summary: 'N/M tests passed' + failing assertions."""
    if not report_path.exists():
        return "(no evaluation file available)"

    lines = report_path.read_text().splitlines()
    n_passed = n_failed = n_total = None
    fails: list[str] = []
    section = None
    cur_fail: list[str] = []
    in_code = False

    def flush_fail() -> None:
        if cur_fail:
            text = " ".join(s.strip() for s in cur_fail if s.strip())
            if text:
                fails.append(text)
            cur_fail.clear()

    for line in lines:
        s = line.strip()
        if s.startswith("─") and "─" in s:
            inner = s.strip("─").strip().lower()
            flush_fail()
            in_code = False
            if "overall stats" in inner:
                section = "stats"
            elif "fails" in inner:
                section = "fails"
            elif "passes" in inner:
                section = "passes"
            else:
                section = None
            continue
        if section == "stats":
            if "Num Passed Tests" in line:
                n_passed = int(line.split(":")[1].strip())
            elif "Num Failed Tests" in line:
                n_failed = int(line.split(":")[1].strip())
            elif "Num Total" in line:
                n_total = int(line.split(":")[1].strip())
            continue
        if section == "fails":
            if s.startswith("```"):
                in_code = not in_code
                continue
            if in_code or s == "None":
                continue
            if s.startswith(">> "):
                flush_fail()
                cur_fail.append(s[3:].strip())
            elif "AssertionError" in s:
                cur_fail.append(s)
            elif s and not s.startswith("-") and cur_fail:
                cur_fail.append(s)
            continue

    flush_fail()
    parts: list[str] = []
    if n_passed is not None and n_total is not None:
        parts.append(f"{n_passed}/{n_total} tests passed.")
    if fails:
        parts.append("Fails:")
        for f in fails:
            parts.append(f"  - {f}")
    return "\n".join(parts) if parts else "(evaluation summary empty)"


# =============================================================================
# Pretty-print structuring helpers (only used when --limit is set)
# =============================================================================

def structure_evaluation_summary(summary: str) -> dict:
    """Parse the compact 'N/M tests passed.\\nFails:\\n  - ...' string into
    {tests_passed, tests_total, fails: [...]} for human-readable pretty output.
    Returns {"raw": summary} if the expected format is not found.
    """
    if not summary or "tests passed" not in summary:
        return {"raw": summary}

    m = re.search(r"(\d+)\s*/\s*(\d+)\s+tests passed", summary)
    if not m:
        return {"raw": summary}

    out: dict = {
        "tests_passed": int(m.group(1)),
        "tests_total": int(m.group(2)),
        "fails": [],
    }
    in_fails = False
    for line in summary.splitlines():
        s = line.rstrip()
        if s.startswith("Fails:"):
            in_fails = True
            continue
        if in_fails:
            stripped = s.lstrip()
            if stripped.startswith("- "):
                out["fails"].append(stripped[2:].strip())
    return out


def structure_trajectory(trajectory: str) -> list[dict]:
    """Parse the formatted '[Step N - AGENT]\\n...\\n[Step N - ENV]\\n...' string
    back into a list of {step, agent, env} dicts for human-readable pretty output.
    """
    if not trajectory:
        return []

    # Split on [Step N - AGENT] / [Step N - ENV] markers
    marker_re = re.compile(r"\[Step\s+(\d+)\s+-\s+(AGENT|ENV)\]\n", re.IGNORECASE)
    parts = marker_re.split(trajectory)
    # parts layout: [pre, "1", "AGENT", body, "1", "ENV", body, "2", "AGENT", body, ...]
    steps: dict[int, dict] = {}
    for i in range(1, len(parts), 3):
        try:
            step_num = int(parts[i])
        except ValueError:
            continue
        role = parts[i + 1].upper()
        body = parts[i + 2].rstrip("\n") if i + 2 < len(parts) else ""
        # Trim a single trailing blank line, keep internal blanks
        body = body.rstrip()
        entry = steps.setdefault(step_num, {"step": step_num, "agent": "", "env": ""})
        if role == "AGENT":
            entry["agent"] = body
        else:
            entry["env"] = body
    return [steps[k] for k in sorted(steps.keys())]


# =============================================================================
# Post-processing & validation
# =============================================================================

def merge_consecutive_same_score(segments: list[dict]) -> list[dict]:
    """Merge adjacent segments sharing both contribution and type."""
    if not segments:
        return segments
    merged = [dict(segments[0])]
    for s in segments[1:]:
        prev = merged[-1]
        if s.get("contribution") == prev.get("contribution") and s.get("type") == prev.get("type"):
            prev["end_step"] = s.get("end_step", prev["end_step"])
            r_prev = prev.get("rationale", "")
            r_new = s.get("rationale", "")
            if r_new and r_new != r_prev:
                prev["rationale"] = f"{r_prev} | {r_new}" if r_prev else r_new
        else:
            merged.append(dict(s))
    return merged


def validate_segments(segments: list) -> tuple[list[dict], Optional[str]]:
    """Filter out invalid segments. Return (valid_segments, error_msg_or_None)."""
    if not isinstance(segments, list):
        return [], f"output is not a list (got {type(segments).__name__})"
    valid: list[dict] = []
    for s in segments:
        if not isinstance(s, dict):
            continue
        try:
            start = int(s["start_step"])
            end = int(s["end_step"])
            contrib = int(s["contribution"])
        except (KeyError, ValueError, TypeError):
            continue
        if start > end or contrib < 1 or contrib > 5:
            continue
        if s.get("type") not in VALID_TYPES:
            continue
        valid.append({
            "start_step": start,
            "end_step": end,
            "subgoal": str(s.get("subgoal", "")),
            "type": s["type"],
            "contribution": contrib,
            "rationale": str(s.get("rationale", "")),
        })
    if not valid:
        return [], "no valid segments after validation"
    return valid, None


# =============================================================================
# OpenAI call
# =============================================================================

def call_joint(
    client: OpenAI,
    prompt: str,
    model: str,
    reasoning_effort: str,
) -> tuple[Optional[list], Optional[dict], Optional[str]]:
    """Call the model. Returns (raw_parsed_segments, usage_dict, error_msg)."""
    last_error = "unknown"
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=reasoning_effort,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            choice = response.choices[0]
            raw = (choice.message.content or "").strip()
            if not raw:
                last_error = f"empty content (finish_reason={choice.finish_reason})"
                raise ValueError(last_error)

            # Strip optional markdown fences
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0].strip()

            segments = json.loads(raw)
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
            return segments, usage, None

        except RateLimitError as e:
            last_error = f"rate_limit: {e}"
            time.sleep(RATE_LIMIT_WAIT * (attempt + 1))
        except (APIError, APITimeoutError) as e:
            last_error = f"api_error: {e}"
            time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
        except json.JSONDecodeError as e:
            last_error = f"json_decode: {e}"
            time.sleep(RETRY_BASE_WAIT ** (attempt + 1))
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            time.sleep(RETRY_BASE_WAIT ** (attempt + 1))

    return None, None, last_error


# =============================================================================
# Per-rollout processor
# =============================================================================

def process_rollout(
    seed: str,
    task_id: str,
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    rollout_root: Path,
) -> dict:
    """Process one (seed, task_id). Returns a dict with status:
    - "ok"      : full record (segments, contributions, cost, etc.)
    - "skipped" : with reason
    - "failed"  : with error
    """
    base = rollout_root / seed / "tasks" / task_id
    env_io_path = base / "logs" / "environment_io.md"
    report_path = base / "evaluation" / "report.md"

    if not env_io_path.exists():
        return {"status": "skipped", "seed": seed, "task_id": task_id, "reason": "no_env_io"}

    size = env_io_path.stat().st_size
    if size > MAX_TRAJECTORY_BYTES:
        return {"status": "skipped", "seed": seed, "task_id": task_id,
                "reason": f"size_exceeded:{size}"}

    instruction = load_instruction(task_id)
    if instruction is None:
        return {"status": "skipped", "seed": seed, "task_id": task_id,
                "reason": "no_instruction"}

    env_io_text = env_io_path.read_text()
    steps = parse_environment_io(env_io_text)

    if not steps:
        return {"status": "skipped", "seed": seed, "task_id": task_id, "reason": "no_steps_parsed"}
    if len(steps) > MAX_STEPS:
        return {"status": "skipped", "seed": seed, "task_id": task_id,
                "reason": f"step_exceeded:{len(steps)}"}

    eval_summary = load_evaluation(report_path)
    trajectory_text = format_trajectory(steps)
    prompt = JOINT_SEGMENT_REWARD_PROMPT.format(
        instruction=instruction,
        trajectory=trajectory_text,
        evaluation_summary=eval_summary,
    )

    t0 = time.time()
    raw_parsed, usage, error = call_joint(client, prompt, model, reasoning_effort)
    elapsed = time.time() - t0

    if raw_parsed is None:
        return {"status": "failed", "seed": seed, "task_id": task_id, "error": error}

    valid_segs, vmsg = validate_segments(raw_parsed)
    if vmsg is not None:
        return {"status": "failed", "seed": seed, "task_id": task_id, "error": vmsg}

    segments = merge_consecutive_same_score(valid_segs)
    contribs = [s["contribution"] for s in segments]

    pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    cost = (
        usage["prompt_tokens"] * pricing["input"]
        + usage["completion_tokens"] * pricing["output"]
    ) / 1_000_000

    return {
        "status": "ok",
        "seed": seed,
        "task_id": task_id,
        "instruction": instruction,
        "evaluation_summary": eval_summary,
        "trajectory": trajectory_text,
        "num_steps": max(s["step"] for s in steps),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "segments": segments,
        "raw_segments": valid_segs,
        "num_segments": len(segments),
        "num_raw_segments": len(valid_segs),
        "mean_contribution": round(sum(contribs) / len(contribs), 3),
        "min_contribution": min(contribs),
        "max_contribution": max(contribs),
        "cost_usd": round(cost, 6),
        "input_tokens": usage["prompt_tokens"],
        "output_tokens": usage["completion_tokens"],
        "elapsed_s": round(elapsed, 1),
    }


# =============================================================================
# Resume / IO helpers
# =============================================================================

def load_already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                done.add(json.loads(line)["task_id"])
            except Exception:
                pass
    return done


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Run only this seed (1-8). Default: all seeds.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max tasks per seed (smoke test).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenAI model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--reasoning", default=DEFAULT_REASONING,
                        choices=["low", "medium", "high"],
                        help=f"reasoning_effort (default: {DEFAULT_REASONING})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cost without calling the API.")
    parser.add_argument("--round", dest="round_name", default=DEFAULT_ROUND,
                        help=f"Rollout generation to score, e.g. round0_9b, round1_9b "
                             f"(default: {DEFAULT_ROUND}). Selects both the source folder "
                             f"under appworld/.../rollout/ and the results subtree under "
                             f"results/rollout/.")
    args = parser.parse_args()

    rollout_root = ROLLOUTS_ROOT / args.round_name
    round_dir = RESULTS_DIR / args.round_name
    variant_dir = round_dir / VARIANT_NAME

    if not rollout_root.exists():
        print(f"ERROR: rollout source does not exist: {rollout_root}", file=sys.stderr)
        sys.exit(1)
    variant_dir.mkdir(parents=True, exist_ok=True)

    seeds = [f"seed_{args.seed}"] if args.seed else [f"seed_{i}" for i in range(1, 9)]

    # Build (seed, task_id) work list, skipping already-done
    work_list: list[tuple[str, str]] = []
    already_done_count = 0
    for seed in seeds:
        seed_dir = rollout_root / seed / "tasks"
        if not seed_dir.exists():
            continue
        task_ids = sorted(p.name for p in seed_dir.iterdir() if p.is_dir())
        if args.limit:
            task_ids = task_ids[:args.limit]
        already = load_already_done(variant_dir / f"{seed}.jsonl")
        for tid in task_ids:
            if tid in already:
                already_done_count += 1
                continue
            work_list.append((seed, tid))

    print(f"Round: {args.round_name}  Variant: {VARIANT_NAME}")
    print(f"Source:  {rollout_root}")
    print(f"Results: {variant_dir}")
    print(f"Model: {args.model}  Reasoning: {args.reasoning}  Workers: {args.workers}")
    print(f"Seeds: {len(seeds)}  Total pending: {len(work_list)}  Already done: {already_done_count}")

    if args.dry_run:
        total_in_chars = 0
        n_skip_size = 0
        for seed, tid in work_list:
            p = rollout_root / seed / "tasks" / tid / "logs" / "environment_io.md"
            if not p.exists():
                continue
            sz = p.stat().st_size
            if sz > MAX_TRAJECTORY_BYTES:
                n_skip_size += 1
            else:
                total_in_chars += sz
        n_run = len(work_list) - n_skip_size
        # Token estimates (4 chars/token, prompt overhead ~3000 tokens, output ~2500 tokens)
        in_tokens = total_in_chars // 4 + 3000 * n_run
        out_tokens = 2500 * n_run
        pricing = MODEL_PRICING.get(args.model, {"input": 0.0, "output": 0.0})
        cost = (in_tokens * pricing["input"] + out_tokens * pricing["output"]) / 1_000_000
        print(f"\n[Dry-run]")
        print(f"  Tasks to call: {n_run}  ({n_skip_size} pre-filtered by size)")
        print(f"  Est. input tokens:  {in_tokens:>12,}")
        print(f"  Est. output tokens: {out_tokens:>12,}")
        print(f"  Est. cost:          ${cost:.2f}")
        return

    # Open output files
    output_files: dict[str, object] = {}
    output_locks: dict[str, threading.Lock] = {}
    for seed in seeds:
        output_files[seed] = open(variant_dir / f"{seed}.jsonl", "a")
        output_locks[seed] = threading.Lock()
    skipped_file = open(round_dir / "skipped.jsonl", "a")
    failed_file = open(round_dir / "failed.jsonl", "a")
    log_lock = threading.Lock()

    # Smoke-test mode: when --limit is set, also dump each successful record
    # as a pretty-printed .json for human spot-checking.
    pretty_dir: Optional[Path] = None
    if args.limit is not None:
        pretty_dir = variant_dir / "pretty"
        pretty_dir.mkdir(parents=True, exist_ok=True)
        print(f"[smoke-mode] Pretty .json per record will be written to: {pretty_dir}")

    state = {
        "done": 0, "ok": 0, "skipped": 0, "failed": 0,
        "total_cost": 0.0, "start_time": time.time(),
    }
    state_lock = threading.Lock()

    client = OpenAI(api_key=OPENAI_API_KEY)

    def worker(item: tuple[str, str]) -> dict:
        seed, tid = item
        try:
            return process_rollout(seed, tid, client, args.model, args.reasoning, rollout_root)
        except Exception as e:
            return {"status": "failed", "seed": seed, "task_id": tid,
                    "error": f"unhandled: {type(e).__name__}: {e}"}

    total = len(work_list)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(worker, item): item for item in work_list}
            for fut in as_completed(futures):
                seed, tid = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {"status": "failed", "seed": seed, "task_id": tid,
                              "error": f"future_error: {e}"}

                status = result.get("status", "unknown")

                with state_lock:
                    state["done"] += 1
                    done = state["done"]
                    if status == "ok":
                        state["ok"] += 1
                        state["total_cost"] += result.get("cost_usd", 0.0)
                    elif status == "skipped":
                        state["skipped"] += 1
                    else:
                        state["failed"] += 1
                    total_cost = state["total_cost"]

                if status == "ok":
                    with output_locks[seed]:
                        f = output_files[seed]
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                    if pretty_dir is not None:
                        # POC-style keys + structured trajectory + structured
                        # evaluation_summary so the segmentation can be
                        # spot-checked against the raw steps the agent took.
                        PRETTY_KEYS = [
                            "task_id", "instruction", "num_steps", "model",
                            "evaluation_summary", "trajectory",
                            "segments", "num_segments",
                            "mean_contribution", "min_contribution", "max_contribution",
                        ]
                        pretty_record = {k: result[k] for k in PRETTY_KEYS if k in result}
                        # Replace the raw strings with structured forms.
                        if "evaluation_summary" in pretty_record:
                            pretty_record["evaluation_summary"] = (
                                structure_evaluation_summary(pretty_record["evaluation_summary"])
                            )
                        if "trajectory" in pretty_record:
                            pretty_record["trajectory"] = (
                                structure_trajectory(pretty_record["trajectory"])
                            )
                        pretty_path = pretty_dir / f"{seed}__{tid}.json"
                        pretty_path.write_text(
                            json.dumps(pretty_record, indent=4, ensure_ascii=False)
                        )
                    print(
                        f"[{seed} {done}/{total}] {tid} ✓ "
                        f"steps={result['num_steps']} segs={result['num_segments']} "
                        f"c_avg={result['mean_contribution']:.2f} "
                        f"cost=${result['cost_usd']:.4f} [{result['elapsed_s']:.1f}s]",
                        flush=True,
                    )
                elif status == "skipped":
                    with log_lock:
                        skipped_file.write(json.dumps({
                            "seed": seed, "task_id": tid, "reason": result["reason"]
                        }) + "\n")
                        skipped_file.flush()
                    print(f"[{seed} {done}/{total}] {tid} ⚠ skipped ({result['reason']})",
                          flush=True)
                else:
                    err = (result.get("error") or "")[:200]
                    with log_lock:
                        failed_file.write(json.dumps({
                            "seed": seed, "task_id": tid, "error": err
                        }) + "\n")
                        failed_file.flush()
                    print(f"[{seed} {done}/{total}] {tid} ✗ failed ({err[:80]})", flush=True)

                # Periodic progress summary
                if done % 10 == 0 or done == total:
                    elapsed = time.time() - state["start_time"]
                    per_task = elapsed / done if done else 0
                    eta_s = per_task * (total - done)
                    print(
                        f"[Progress] {done}/{total}  "
                        f"ok={state['ok']} skip={state['skipped']} fail={state['failed']}  "
                        f"cost=${total_cost:.2f}  "
                        f"elapsed={elapsed/60:.1f}m  ETA={eta_s/60:.1f}m",
                        flush=True,
                    )

    finally:
        for f in output_files.values():
            f.close()
        skipped_file.close()
        failed_file.close()

    elapsed = time.time() - state["start_time"]
    print(f"\n=== Final ===")
    print(f"Done:        {state['done']}/{total}")
    print(f"OK:          {state['ok']}")
    print(f"Skipped:     {state['skipped']}")
    print(f"Failed:      {state['failed']}")
    print(f"Total cost:  ${state['total_cost']:.2f}")
    print(f"Total time:  {elapsed/60:.1f}m")


if __name__ == "__main__":
    main()
