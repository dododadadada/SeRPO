"""Shared prompt-building helpers for KS_baseline rubric scoring.

Factored out of run_rollouts_KS_baseline.py so that grpo_online/judge.py
(and future scripts) can import ``build_ks_baseline_prompt`` without
depending on the full run_rollouts_* CLI machinery.

run_rollouts_KS_baseline.py behaviour is unchanged: it still defines its
own copies of the helpers, so this module is additive/non-invasive.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

REPO = Path("/data/minjeong/Autonomous_agent")
BASE_TASKS_DIR = REPO / "rubric_reward_poc(mid_report)" / "data" / "tasks"

# ---------------------------------------------------------------------------
# Prompt template (verbatim from run_rollouts_KS_baseline.py)
# ---------------------------------------------------------------------------

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

### Anchoring principles
- **Score each segment on its own outcome**, not on whether earlier segments were clean. A successful recovery is genuinely valuable and earns a high score.
- A "wasted" segment can still get 2-3 if it produced information that was directly used in the next segment (i.e., the error message led to recovery).
- A *failed* attempt — regardless of whether it's the first try or a recovery attempt — scores 1-2. If the agent tries to recover and the recovery itself also fails, that recovery segment is 1-2.
- A *successful* recovery — agent failed earlier, then found the right approach and made progress — scores on its own merits (typically 4-5 if the recovery was clean and effective).
- The PRIOR segment that caused the need for recovery stays at 1-2. We do not retroactively penalize successful recoveries for the existence of an earlier mistake; we already penalize the mistake itself in its own segment.
- The final `completion` segment: 5 if the answer is correct (per evaluation result), 1 if wrong, 3 if a near-miss (mostly-correct answer or partial pass).
- A `complete_task(status="fail")` call — or any other "I give up" completion that does not deliver the task goal — is outcome score **1**, regardless of whether the API call itself executed successfully. Score the completion segment by whether the task goal was achieved, not by whether the completion API was invoked without error. The agent submitting "fail" is the agent reporting its own failure; do not reward graceful surrender.

### Common failure pattern to attribute correctly
The most common AppWorld failure: agent hallucinates an API name (`get_contacts`, `list_notes`, `list_files`) and crashes.
- The CRASHING segment gets contribution **1** (premature action, no exploration).
- The subsequent recovery segment that finds the right API and makes progress gets **4-5** — it did real productive work, even if it wouldn't have been needed without the earlier mistake.
- If the recovery ALSO fails (e.g., agent guesses another wrong API after the first crash), that recovery segment is **1-2**.

---

## Output format

Respond ONLY with a JSON array. No markdown fences, no extra prose.

[
  {{"start_step": <int>, "end_step": <int>, "subgoal": "<3–8 words>", "type": "<one of the 7 types>", "contribution": <1-5>, "rationale": "<one short sentence, <25 words>"}},
  ...
]
"""

# ---------------------------------------------------------------------------
# Trajectory parsers (mirrored from run_rollouts_KS_baseline.py)
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"###\s*Environment Interaction\s*(\d+)", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)


def _parse_environment_io(text: str) -> list[dict]:
    """Parse environment_io.md into [{step, agent, env}, ...]."""
    sections = _SECTION_RE.split(text)
    steps: list[dict] = []
    for i in range(1, len(sections), 2):
        try:
            step_num = int(sections[i])
        except ValueError:
            continue
        body = sections[i + 1] if i + 1 < len(sections) else ""
        blocks = _CODE_BLOCK_RE.findall(body)
        agent_code = blocks[0][1].strip() if blocks else ""
        env_output = blocks[1][1].strip() if len(blocks) >= 2 else ""
        steps.append({"step": step_num, "agent": agent_code, "env": env_output})
    return steps


def _format_trajectory(steps: list[dict]) -> str:
    lines: list[str] = []
    for s in steps:
        lines.append(f"[Step {s['step']} - AGENT]\n{s['agent']}")
        if s["env"]:
            lines.append(f"[Step {s['step']} - ENV]\n{s['env']}")
    return "\n\n".join(lines)


def _load_instruction(task_id: str) -> Optional[str]:
    p = BASE_TASKS_DIR / task_id / "instruction.txt"
    if not p.exists():
        return None
    return p.read_text().strip()


def _load_evaluation(report_path: Path) -> str:
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ks_baseline_prompt(lm_calls_path: str) -> list[dict]:
    """Build the KS_baseline rubric messages for one trajectory.

    Args:
        lm_calls_path: path to the trajectory's lm_calls.jsonl file, e.g.
            ``.../rollout/round0_9b/seed_1/tasks/<tid>/logs/lm_calls.jsonl``.
            The task directory is inferred as two levels up (logs/ -> task/).

    Returns:
        A single-element list suitable for ``client.chat.completions.create``
        (role=user, content=formatted prompt).

    Raises:
        FileNotFoundError: if environment_io.md is missing.
        ValueError: if no steps can be parsed or instruction is missing.
    """
    lm_calls_path = Path(lm_calls_path)
    task_dir = lm_calls_path.parent.parent  # .../tasks/<task_id>/
    task_id = task_dir.name

    env_io_path = task_dir / "logs" / "environment_io.md"
    if not env_io_path.exists():
        raise FileNotFoundError(f"environment_io.md not found: {env_io_path}")

    report_path = task_dir / "evaluation" / "report.md"

    instruction = _load_instruction(task_id)
    if instruction is None:
        raise ValueError(f"instruction.txt not found for task_id={task_id!r}")

    env_io_text = env_io_path.read_text()
    steps = _parse_environment_io(env_io_text)
    if not steps:
        raise ValueError(f"no steps parsed from {env_io_path}")

    eval_summary = _load_evaluation(report_path)
    trajectory_text = _format_trajectory(steps)

    prompt_text = JOINT_SEGMENT_REWARD_PROMPT.format(
        instruction=instruction,
        trajectory=trajectory_text,
        evaluation_summary=eval_summary,
    )
    return [{"role": "user", "content": prompt_text}]
