# GRPO v1-claim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 3 GRPO trainer producing two Round-1 LoRA checkpoints (Vanilla GRPO + SeRPO) on the Fail-Only condition (81 tasks), evaluated on AppWorld test_normal.

**Architecture:** Offline-only verl-based trainer. Preprocessing converts `(lm_calls.jsonl, joint_seed_N.jsonl, train.json)` into per-method parquet files with precomputed per-token advantage tensors. verl `RayPPOTrainer` is subclassed to skip rollout + reward computation; advantages are read directly from the parquet. Ref policy via `peft.disable_adapter()` (no second model instance). External vLLM remains for ref logprob caching utility and final test_normal eval (LoRA hot-swap).

**Tech Stack:** Python 3.11 in conda env `appworld`. torch 2.10 + CUDA 12.8 + transformers 4.57. To install: verl (>=0.5), peft, datasets, pyarrow, accelerate. Existing infra: external vLLM via `serve_qwen_vllm.sh`, AppWorld native test runner.

**Spec:** `docs/superpowers/specs/2026-05-13-grpo-v1-claim-design.md`

---

## File map

```
grpo/
├── pyproject.toml                          # package metadata + deps
├── __init__.py
├── preprocess/
│   ├── __init__.py
│   ├── failset.py                          # Task 1
│   ├── tokenize_trajectory.py              # Task 2
│   ├── make_advantages.py                  # Task 3
│   └── build_dataset.py                    # Task 4
├── trainer/
│   ├── __init__.py
│   ├── config/
│   │   ├── base_round1.yaml                # verl shared config       # Task 5
│   │   ├── vanilla_round1.yaml             # overrides for vanilla    # Task 5
│   │   └── serpo_round1.yaml               # overrides for serpo      # Task 5
│   ├── dataset.py                          # ParquetGRPODataset        # Task 5
│   ├── offline_trainer.py                  # OfflineRayPPOTrainer      # Task 5
│   └── run_grpo.py                         # Hydra entrypoint          # Task 5
├── eval/
│   ├── __init__.py
│   └── run_test_normal.py                  # Task 6
├── scripts/
│   ├── train_vanilla_round1.sh             # Task 7
│   └── train_serpo_round1.sh               # Task 7
└── tests/
    ├── __init__.py
    ├── fixtures/                           # 1 sample task copied from seed_1
    │   ├── seed_1_07b42fd_1_lm_calls.jsonl
    │   └── joint_07b42fd_1.json
    ├── test_failset.py                     # Task 1
    ├── test_tokenize_trajectory.py         # Task 2
    ├── test_make_advantages.py             # Task 3
    └── test_build_dataset.py               # Task 4
```

Build artifacts (gitignore): `grpo/data/`, `grpo/ckpts/`, `grpo/metrics/`, `grpo/tests/fixtures/*.jsonl` (large).

---

## Task 0: Environment + scaffolding

**Files:**
- Create: `grpo/pyproject.toml`, `grpo/__init__.py`, `grpo/.gitignore`
- Create directories: `grpo/{preprocess,trainer,trainer/config,eval,scripts,tests,tests/fixtures}/__init__.py`
- Modify: none yet

- [ ] **Step 0.1: Confirm conda env active**

```bash
echo "$CONDA_DEFAULT_ENV"
```
Expected: `appworld`. If not, `conda activate appworld` first.

- [ ] **Step 0.2: Install missing dependencies**

```bash
pip install "verl>=0.5" peft datasets pyarrow accelerate hydra-core
```
Expected: installed successfully. If verl install fails due to Ray version conflicts, pin with `pip install "ray[default]>=2.31" verl peft datasets pyarrow accelerate hydra-core`.

- [ ] **Step 0.3: Verify imports**

```bash
python -c "import verl, peft, datasets, pyarrow, accelerate, transformers, torch; print('ok')"
```
Expected: `ok` printed.

- [ ] **Step 0.4: Initialize git (if not already)**

```bash
cd /data/minjeong/Autonomous_agent && git init && git config user.email "noreply@anthropic.com" && git config user.name "minjeong"
```
Expected: `Initialized empty Git repository`. (Skip if already initialized.)

- [ ] **Step 0.5: Create root .gitignore**

Write `/data/minjeong/Autonomous_agent/.gitignore`:

```
# python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.ruff_cache/

# environments
.env
*.venv/

# appworld artifacts (large)
appworld/experiments/outputs/

# grpo artifacts
grpo/data/
grpo/ckpts/
grpo/metrics/
grpo/tests/fixtures/*.jsonl
grpo/tests/fixtures/*.json

# rubric reward artifacts
rubric_reward/results/
rubric_reward_poc(mid_report)/results/

# misc
*.log
```

- [ ] **Step 0.6: Create grpo package scaffolding**

Run:
```bash
cd /data/minjeong/Autonomous_agent
mkdir -p grpo/{preprocess,trainer/config,eval,scripts,tests/fixtures,data,ckpts,metrics}
touch grpo/__init__.py grpo/preprocess/__init__.py grpo/trainer/__init__.py grpo/eval/__init__.py grpo/tests/__init__.py
```

- [ ] **Step 0.7: Write `grpo/pyproject.toml`**

```toml
[project]
name = "grpo"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.10",
    "transformers>=4.57",
    "verl>=0.5",
    "peft>=0.13",
    "datasets>=2.20",
    "pyarrow>=15.0",
    "accelerate>=0.34",
    "hydra-core>=1.3",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["grpo*"]
```

- [ ] **Step 0.8: Install grpo package in editable mode**

```bash
cd /data/minjeong/Autonomous_agent/grpo && pip install -e .
```
Expected: `Successfully installed grpo-0.1.0`.

- [ ] **Step 0.9: First commit**

```bash
cd /data/minjeong/Autonomous_agent && git add docs/ grpo/__init__.py grpo/pyproject.toml grpo/preprocess/__init__.py grpo/trainer/__init__.py grpo/eval/__init__.py grpo/tests/__init__.py .gitignore && git commit -m "scaffold: grpo package, design + plan docs"
```

---

## Task 1: `failset.py` — extract strict 0/8 fail task_ids

**Files:**
- Create: `grpo/preprocess/failset.py`
- Test: `grpo/tests/test_failset.py`

- [ ] **Step 1.1: Write test `tests/test_failset.py`**

```python
"""Tests for grpo.preprocess.failset."""
import json
from pathlib import Path
import tempfile
import pytest

from grpo.preprocess.failset import build_failset


def _make_seed_eval(tmpdir: Path, seed: int, task_outcomes: dict[str, bool]) -> Path:
    """Write a fake seed_N/evaluations/train.json mimicking AppWorld format."""
    seed_dir = tmpdir / f"seed_{seed}" / "evaluations"
    seed_dir.mkdir(parents=True)
    payload = {
        "aggregate": {"task_goal_completion": 0.0, "scenario_goal_completion": 0.0},
        "individual": {
            tid: {"success": ok, "difficulty": 1, "num_tests": 2, "passes": [], "failures": []}
            for tid, ok in task_outcomes.items()
        },
    }
    (seed_dir / "train.json").write_text(json.dumps(payload))
    return tmpdir


def test_failset_strict_zero_of_eight(tmp_path: Path) -> None:
    """Returns only task_ids that fail in ALL 8 seeds."""
    base = tmp_path / "rollout" / "round0"
    base.mkdir(parents=True)
    # task_A: 0/8 success → in failset
    # task_B: 1/8 success (seed_3 succeeds) → not in failset
    # task_C: 0/8 success → in failset
    for s in range(1, 9):
        _make_seed_eval(
            base,
            s,
            {
                "task_A": False,
                "task_B": (s == 3),
                "task_C": False,
            },
        )
    failset = build_failset(base)
    assert sorted(failset) == ["task_A", "task_C"]


def test_failset_missing_seed_raises(tmp_path: Path) -> None:
    """If fewer than 8 seed dirs are present, raise."""
    base = tmp_path / "rollout" / "round0"
    base.mkdir(parents=True)
    for s in (1, 2, 3):  # only 3 seeds
        _make_seed_eval(base, s, {"task_A": False})
    with pytest.raises(ValueError, match="expected 8 seeds"):
        build_failset(base)


def test_failset_real_data() -> None:
    """End-to-end test against real Round-0 data: expect exactly 81 task_ids."""
    base = Path("/data/minjeong/Autonomous_agent/appworld/experiments/outputs/rollout/round0")
    if not base.exists():
        pytest.skip("Round-0 rollout not present")
    failset = build_failset(base)
    assert len(failset) == 81
```

- [ ] **Step 1.2: Run test, expect ImportError**

```bash
cd /data/minjeong/Autonomous_agent && python -m pytest grpo/tests/test_failset.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'grpo.preprocess.failset'`.

- [ ] **Step 1.3: Implement `grpo/preprocess/failset.py`**

```python
"""Extract strict 0/8 fail-only task_id set from Round-0 evaluations."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path


def build_failset(rollout_round_dir: Path, n_seeds_expected: int = 8) -> list[str]:
    """Return task_ids that fail in every seed under ``rollout_round_dir``.

    Expects layout: ``rollout_round_dir/seed_{1..N}/evaluations/train.json`` where
    each train.json has ``individual[task_id]["success"]: bool``.
    """
    rollout_round_dir = Path(rollout_round_dir)
    seed_dirs = sorted(rollout_round_dir.glob("seed_*"))
    if len(seed_dirs) != n_seeds_expected:
        raise ValueError(
            f"expected 8 seeds under {rollout_round_dir}, found {len(seed_dirs)}: {seed_dirs}"
        )

    success_count: dict[str, int] = defaultdict(int)
    all_tids: set[str] = set()
    for seed_dir in seed_dirs:
        eval_path = seed_dir / "evaluations" / "train.json"
        with eval_path.open() as f:
            data = json.load(f)
        for tid, rec in data["individual"].items():
            all_tids.add(tid)
            if rec.get("success", False):
                success_count[tid] += 1

    return sorted(tid for tid in all_tids if success_count[tid] == 0)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    failset = build_failset(args.rollout_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(failset, indent=2))
    print(f"wrote {len(failset)} task_ids to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.4: Run tests, expect PASS**

```bash
cd /data/minjeong/Autonomous_agent && python -m pytest grpo/tests/test_failset.py -v
```
Expected: 3 passed (or 2 passed + 1 skipped if real data unavailable).

- [ ] **Step 1.5: Commit**

```bash
git add grpo/preprocess/failset.py grpo/tests/test_failset.py && git commit -m "feat(preprocess): strict 0/8 fail-only task_id extraction"
```

---

## Task 2: `tokenize_trajectory.py` — cumulative messages → per-step token ranges

**Files:**
- Create: `grpo/preprocess/tokenize_trajectory.py`
- Test: `grpo/tests/test_tokenize_trajectory.py`
- Fixture: `grpo/tests/fixtures/seed_1_07b42fd_1_lm_calls.jsonl` (copied from real rollout)

- [ ] **Step 2.1: Copy fixture trajectory**

```bash
cp /data/minjeong/Autonomous_agent/appworld/experiments/outputs/rollout/round0/seed_1/tasks/07b42fd_1/logs/lm_calls.jsonl \
   /data/minjeong/Autonomous_agent/grpo/tests/fixtures/seed_1_07b42fd_1_lm_calls.jsonl
```

- [ ] **Step 2.2: Write test `tests/test_tokenize_trajectory.py`**

```python
"""Tests for grpo.preprocess.tokenize_trajectory."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from transformers import AutoTokenizer

from grpo.preprocess.tokenize_trajectory import tokenize_trajectory, PREFIX_MARKER

FIXTURE = Path(__file__).parent / "fixtures" / "seed_1_07b42fd_1_lm_calls.jsonl"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


def test_tokenize_returns_expected_shape(tokenizer):
    result = tokenize_trajectory(FIXTURE, tokenizer)
    assert "input_ids" in result
    assert "response_mask" in result
    assert "step_token_ranges" in result
    assert len(result["input_ids"]) == len(result["response_mask"])
    assert all(m in (0, 1) for m in result["response_mask"])


def test_prefix_marker_found(tokenizer):
    """Decoded prefix should contain the marker; response_mask=0 over prefix."""
    result = tokenize_trajectory(FIXTURE, tokenizer)
    # First step assistant tokens start AFTER prefix → some leading 0s in response_mask
    assert result["response_mask"][0] == 0
    # At least one assistant token after the prefix
    assert sum(result["response_mask"]) > 0


def test_step_ranges_cover_assistant_tokens(tokenizer):
    """Token ranges from step_token_ranges should each map to response_mask=1 spans."""
    result = tokenize_trajectory(FIXTURE, tokenizer)
    mask = result["response_mask"]
    for start, end in result["step_token_ranges"]:
        assert end > start, f"empty step range: ({start}, {end})"
        assert end <= len(mask)
        # Every token in this range should be a response token
        assert all(mask[i] == 1 for i in range(start, end)), \
            f"step range [{start}:{end}] contains non-response tokens"


def test_step_count_matches_lm_calls(tokenizer):
    """Number of steps should equal the number of (asst, user) pairs after prefix."""
    result = tokenize_trajectory(FIXTURE, tokenizer)
    # The fixture trajectory has 6 lm_calls → 6 assistant steps
    with FIXTURE.open() as f:
        n_calls = sum(1 for _ in f)
    assert len(result["step_token_ranges"]) == n_calls


def test_decoded_step_matches_original(tokenizer):
    """Decoding tokens of step 1 should yield text containing some original assistant content."""
    result = tokenize_trajectory(FIXTURE, tokenizer)
    s0, e0 = result["step_token_ranges"][0]
    decoded = tokenizer.decode(result["input_ids"][s0:e0])
    # Step 1 of 07b42fd_1 mentions Spotify login
    assert "spotify" in decoded.lower() or "supervisor" in decoded.lower()


def test_missing_marker_raises(tokenizer, tmp_path):
    """If the prefix marker is missing, raise ValueError."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({
        "input": {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]}
    }) + "\n")
    with pytest.raises(ValueError, match="prefix marker"):
        tokenize_trajectory(bad, tokenizer)
```

- [ ] **Step 2.3: Run test, expect ImportError**

```bash
cd /data/minjeong/Autonomous_agent && python -m pytest grpo/tests/test_tokenize_trajectory.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 2.4: Implement `grpo/preprocess/tokenize_trajectory.py`**

```python
"""Tokenize an AppWorld trajectory and produce per-step assistant token ranges.

Input:  appworld/.../seed_N/tasks/<tid>/logs/lm_calls.jsonl
Output: {input_ids, attention_mask, response_mask, step_token_ranges, num_steps}

Mask rule:
  - System / few-shot / task-instruction tokens: response_mask = 0
  - Step k assistant message tokens (k = 1..N): response_mask = 1
  - Step k env-output user message tokens: response_mask = 0
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

PREFIX_MARKER = "Using these APIs, now generate code to solve the actual task:"


def _read_last_messages(lm_calls_path: Path) -> list[dict[str, str]]:
    """Read the messages list from the LAST line (cumulative)."""
    last: str | None = None
    with lm_calls_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        raise ValueError(f"empty lm_calls file: {lm_calls_path}")
    rec = json.loads(last)
    msgs = rec.get("input", {}).get("messages")
    if not msgs:
        raise ValueError(f"no input.messages in last line of {lm_calls_path}")
    return msgs


def _find_prefix_end(messages: list[dict[str, str]]) -> int:
    """Return the index (inclusive) of the last prefix message — the user message
    whose content ENDS WITH PREFIX_MARKER (followed by the actual task instruction)."""
    for i in reversed(range(len(messages))):
        m = messages[i]
        if m.get("role") == "user" and PREFIX_MARKER in m.get("content", ""):
            return i
    raise ValueError(
        f"prefix marker not found in messages "
        f"(expected substring: {PREFIX_MARKER!r})"
    )


def tokenize_trajectory(lm_calls_path: Path, tokenizer) -> dict[str, Any]:
    """Tokenize one trajectory and produce per-step assistant token ranges.

    Returns dict with keys:
      input_ids:      list[int]
      attention_mask: list[int]   (all 1)
      response_mask:  list[int]   (1 on assistant tokens of steps 1..N only)
      step_token_ranges: list[(start, end)]  (length = num_steps)
      num_steps:      int
    """
    lm_calls_path = Path(lm_calls_path)
    messages = _read_last_messages(lm_calls_path)
    prefix_end = _find_prefix_end(messages)
    interaction = messages[prefix_end + 1 :]

    # Validate: interaction starts with assistant (step 1)
    if not interaction or interaction[0].get("role") != "assistant":
        raise ValueError(
            f"expected first interaction message to be assistant (step 1), "
            f"got role={interaction[0].get('role') if interaction else None!r}"
        )

    # Tokenize step-by-step so we can record exact token ranges.
    # Strategy: tokenize the cumulative prefix, then for each step (asst, user) pair,
    # tokenize asst alone to get its length, then user alone.
    # We use apply_chat_template per-message with add_generation_prompt=False and
    # tokenize=True; concatenation matches a single apply_chat_template call.

    # Compute the prefix token sequence (prefix_end+1 messages, ending with the
    # task instruction user message). add_generation_prompt=False keeps it as
    # a closed prefix; the assistant turn that follows starts a new role tag.
    prefix_tokens = tokenizer.apply_chat_template(
        messages[: prefix_end + 1],
        tokenize=True,
        add_generation_prompt=False,
    )
    input_ids: list[int] = list(prefix_tokens)
    response_mask: list[int] = [0] * len(prefix_tokens)
    step_ranges: list[tuple[int, int]] = []

    # Walk the interaction. For each assistant message, append its tokens and mark
    # response_mask=1 on those tokens (including the role-tag tokens; this is the
    # standard convention in verl/trl multi-turn loss).
    cumulative = list(messages[: prefix_end + 1])
    for msg in interaction:
        cumulative.append(msg)
        new_full = tokenizer.apply_chat_template(
            cumulative,
            tokenize=True,
            add_generation_prompt=False,
        )
        added_start = len(input_ids)
        added_end = len(new_full)
        # Sanity: prefix of new_full must equal current input_ids.
        if new_full[:added_start] != input_ids:
            raise RuntimeError(
                "chat template tokenization is not append-only; cannot align step ranges. "
                f"Diff at index {next(i for i,(a,b) in enumerate(zip(new_full, input_ids)) if a!=b)}"
            )
        input_ids = list(new_full)
        is_asst = msg.get("role") == "assistant"
        response_mask.extend([1 if is_asst else 0] * (added_end - added_start))
        if is_asst:
            step_ranges.append((added_start, added_end))

    assert len(input_ids) == len(response_mask)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "response_mask": response_mask,
        "step_token_ranges": step_ranges,
        "num_steps": len(step_ranges),
    }
```

- [ ] **Step 2.5: Run tests, expect PASS**

```bash
cd /data/minjeong/Autonomous_agent && python -m pytest grpo/tests/test_tokenize_trajectory.py -v
```
Expected: 6 passed.

- [ ] **Step 2.6: Sanity check on a real trajectory (manual)**

```bash
python - <<'PY'
from pathlib import Path
from transformers import AutoTokenizer
from grpo.preprocess.tokenize_trajectory import tokenize_trajectory
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
r = tokenize_trajectory(Path("/data/minjeong/Autonomous_agent/appworld/experiments/outputs/rollout/round0/seed_1/tasks/07b42fd_1/logs/lm_calls.jsonl"), tok)
print("len:", len(r["input_ids"]), "num_steps:", r["num_steps"], "resp_ratio:", sum(r["response_mask"])/len(r["response_mask"]))
PY
```
Expected: response ratio in roughly 5-30% range; num_steps = 6 (for 07b42fd_1).

- [ ] **Step 2.7: Commit**

```bash
git add grpo/preprocess/tokenize_trajectory.py grpo/tests/test_tokenize_trajectory.py && git commit -m "feat(preprocess): tokenize trajectory with per-step assistant token ranges"
```

---

## Task 3: `make_advantages.py` — segments/outcomes → per-token advantage tensor

**Files:**
- Create: `grpo/preprocess/make_advantages.py`
- Test: `grpo/tests/test_make_advantages.py`

- [ ] **Step 3.1: Write test `tests/test_make_advantages.py`**

```python
"""Tests for grpo.preprocess.make_advantages."""
from __future__ import annotations
import numpy as np
import pytest

from grpo.preprocess.make_advantages import (
    compute_vanilla_advantage,
    compute_serpo_advantage,
    RolloutData,
)


def _mock_rollout(num_tokens: int, response_mask: list[int],
                  step_token_ranges: list[tuple[int, int]],
                  segments: list[dict], outcome: int) -> RolloutData:
    return RolloutData(
        task_id="t", seed=1,
        input_ids=list(range(num_tokens)),
        attention_mask=[1] * num_tokens,
        response_mask=list(response_mask),
        step_token_ranges=list(step_token_ranges),
        segments=list(segments),
        outcome=outcome,
    )


def test_vanilla_all_zero_outcomes_yields_zero_advantage():
    """In fail-only set, all 8 outcomes = 0 → advantage scalar = 0 → all-zero tensor."""
    rollouts = [
        _mock_rollout(10, [0,0,1,1,1,0,1,1,0,0], [(2,5),(6,8)], [], 0)
        for _ in range(8)
    ]
    compute_vanilla_advantage(rollouts)
    for r in rollouts:
        assert np.allclose(r.token_adv, 0.0)


def test_vanilla_mixed_outcomes_has_signal():
    """Mixed outcomes → nonzero advantage on assistant tokens, zero on others."""
    rollouts = []
    for i in range(8):
        outcome = 1 if i < 2 else 0  # 2 successes
        rollouts.append(_mock_rollout(10, [0,0,1,1,1,0,1,1,0,0], [(2,5),(6,8)], [], outcome))
    compute_vanilla_advantage(rollouts)
    # Tokens at positions where response_mask=0 stay 0
    for r in rollouts:
        for i, m in enumerate(r.response_mask):
            if m == 0:
                assert r.token_adv[i] == 0.0
    # Assistant token advantage is the rollout-level scalar, identical across tokens
    for r in rollouts:
        active = [r.token_adv[i] for i, m in enumerate(r.response_mask) if m == 1]
        assert all(v == active[0] for v in active)
    # Group mean of trajectory-level scalars ≈ 0
    scalars = [
        next(r.token_adv[i] for i, m in enumerate(r.response_mask) if m == 1)
        for r in rollouts
    ]
    assert abs(float(np.mean(scalars))) < 1e-5


def test_serpo_segment_broadcast():
    """SeRPO: each segment's tokens get the same z-scored contribution."""
    # 1 rollout, 2 steps, 2 segments (each segment covers 1 step)
    rollouts = []
    for i in range(8):
        contrib = (i % 5) + 1  # 1..5 distributed
        segs = [
            {"start_step": 1, "end_step": 1, "contribution": contrib},
            {"start_step": 2, "end_step": 2, "contribution": 6 - contrib},
        ]
        # response_mask: step1 tokens at [2:5], step2 tokens at [6:8]
        rollouts.append(_mock_rollout(10, [0,0,1,1,1,0,1,1,0,0], [(2,5),(6,8)], segs, 0))
    compute_serpo_advantage(rollouts)
    # Each segment's tokens have identical advantage
    for r in rollouts:
        for seg in r.segments:
            tok_start, tok_end = r.step_token_ranges[seg["start_step"] - 1]
            vals = r.token_adv[tok_start:tok_end]
            assert np.allclose(vals, vals[0])
    # Group mean over all assistant tokens ≈ 0 (z-score property)
    all_active_vals = np.concatenate([
        r.token_adv[np.array(r.response_mask, dtype=bool)] for r in rollouts
    ])
    assert abs(float(all_active_vals.mean())) < 1e-5


def test_serpo_zero_outside_assistant():
    """Tokens with response_mask=0 stay at zero."""
    rollouts = []
    for i in range(8):
        segs = [{"start_step": 1, "end_step": 1, "contribution": i % 5 + 1}]
        rollouts.append(_mock_rollout(10, [0,0,1,1,1,0,0,0,0,0], [(2,5)], segs, 0))
    compute_serpo_advantage(rollouts)
    for r in rollouts:
        for i, m in enumerate(r.response_mask):
            if m == 0:
                assert r.token_adv[i] == 0.0


def test_serpo_steps_in_segmentation_gap_stay_zero():
    """Steps not covered by any segment → token_adv = 0."""
    rollouts = []
    for i in range(8):
        # 2 steps but segment only covers step 1
        segs = [{"start_step": 1, "end_step": 1, "contribution": (i % 5) + 1}]
        rollouts.append(_mock_rollout(10, [0,0,1,1,1,0,1,1,0,0], [(2,5),(6,8)], segs, 0))
    compute_serpo_advantage(rollouts)
    # Step 2 tokens [6:8] should remain 0
    for r in rollouts:
        assert np.all(r.token_adv[6:8] == 0.0)
```

- [ ] **Step 3.2: Run test, expect ImportError**

```bash
cd /data/minjeong/Autonomous_agent && python -m pytest grpo/tests/test_make_advantages.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3.3: Implement `grpo/preprocess/make_advantages.py`**

```python
"""Compute per-token advantage tensors for Vanilla GRPO and SeRPO.

Both operate on a per-task group of 8 rollouts. The advantage is broadcast onto
each rollout's pre-existing token layout (assistant token spans).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np

EPS = 1e-8


@dataclass
class RolloutData:
    """One rollout's preprocessed view, mutated in-place by the advantage funcs."""
    task_id: str
    seed: int
    input_ids: list[int]
    attention_mask: list[int]
    response_mask: list[int]
    step_token_ranges: list[tuple[int, int]]
    segments: list[dict[str, Any]]
    outcome: int
    token_adv: np.ndarray = field(default=None)  # populated by compute_*

    def __post_init__(self) -> None:
        if self.token_adv is None:
            self.token_adv = np.zeros(len(self.input_ids), dtype=np.float32)


def compute_vanilla_advantage(rollouts: list[RolloutData]) -> None:
    """Outcome-only z-score, broadcast trajectory-uniformly to assistant tokens."""
    outcomes = np.asarray([r.outcome for r in rollouts], dtype=np.float32)
    mu = float(outcomes.mean())
    sigma = float(outcomes.std()) + EPS
    for r in rollouts:
        scalar = (float(r.outcome) - mu) / sigma
        mask = np.asarray(r.response_mask, dtype=bool)
        r.token_adv = np.zeros(len(r.input_ids), dtype=np.float32)
        r.token_adv[mask] = scalar


def compute_serpo_advantage(rollouts: list[RolloutData]) -> None:
    """Per-segment z-score over the pooled contributions of all 8 rollouts;
    broadcast each segment's z-score to its assistant token span."""
    pool = np.asarray(
        [seg["contribution"] for r in rollouts for seg in r.segments],
        dtype=np.float32,
    )
    if pool.size == 0:
        for r in rollouts:
            r.token_adv = np.zeros(len(r.input_ids), dtype=np.float32)
        return
    mu = float(pool.mean())
    sigma = float(pool.std()) + EPS
    for r in rollouts:
        r.token_adv = np.zeros(len(r.input_ids), dtype=np.float32)
        for seg in r.segments:
            a_hat = (float(seg["contribution"]) - mu) / sigma
            start_step = int(seg["start_step"])
            end_step = int(seg["end_step"])
            for k in range(start_step, end_step + 1):
                idx = k - 1
                if idx < 0 or idx >= len(r.step_token_ranges):
                    continue  # segment refers to nonexistent step
                tok_start, tok_end = r.step_token_ranges[idx]
                r.token_adv[tok_start:tok_end] = a_hat
```

- [ ] **Step 3.4: Run tests, expect PASS**

```bash
cd /data/minjeong/Autonomous_agent && python -m pytest grpo/tests/test_make_advantages.py -v
```
Expected: 5 passed.

- [ ] **Step 3.5: Commit**

```bash
git add grpo/preprocess/make_advantages.py grpo/tests/test_make_advantages.py && git commit -m "feat(preprocess): per-token advantage for vanilla GRPO and SeRPO"
```

---

## Task 4: `build_dataset.py` — orchestrate per-method parquet build

**Files:**
- Create: `grpo/preprocess/build_dataset.py`
- Test: `grpo/tests/test_build_dataset.py`
- Fixture: `grpo/tests/fixtures/joint_07b42fd_1.json` (built in Step 4.1)

- [ ] **Step 4.1: Build a fixture joint-reward file for one task**

Since Phase 2 hasn't run yet, fabricate a minimal joint reward record for the fixture task. Write `grpo/tests/fixtures/joint_07b42fd_1.json`:

```json
{
  "task_id": "07b42fd_1",
  "instruction": "Follow all the classical artists on Spotify that have at least 22 followers.",
  "num_steps": 6,
  "segments": [
    {"start_step": 1, "end_step": 1, "subgoal": "login", "type": "login", "contribution": 5, "rationale": "ok"},
    {"start_step": 2, "end_step": 3, "subgoal": "explore", "type": "api_exploration", "contribution": 4, "rationale": "ok"},
    {"start_step": 4, "end_step": 4, "subgoal": "follow", "type": "action", "contribution": 5, "rationale": "ok"},
    {"start_step": 5, "end_step": 6, "subgoal": "report", "type": "completion", "contribution": 3, "rationale": "near-miss"}
  ],
  "raw_segments": [],
  "num_segments": 4,
  "mean_contribution": 4.25
}
```

- [ ] **Step 4.2: Write test `tests/test_build_dataset.py`**

```python
"""Integration test for build_dataset orchestrator."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
import pyarrow.parquet as pq

from grpo.preprocess.build_dataset import build_dataset_for_task_group


FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_LM_CALLS = FIXTURE_DIR / "seed_1_07b42fd_1_lm_calls.jsonl"
FIXTURE_JOINT = FIXTURE_DIR / "joint_07b42fd_1.json"


def test_build_dataset_one_task_serpo(tmp_path):
    """End-to-end: 1 task × 1 seed (copy fixture 8 times to simulate group) → parquet."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    joint = json.loads(FIXTURE_JOINT.read_text())
    # Build 8 fake rollouts by reusing the same lm_calls + joint segments
    inputs = []
    for seed in range(1, 9):
        inputs.append({
            "task_id": "07b42fd_1",
            "seed": seed,
            "lm_calls_path": FIXTURE_LM_CALLS,
            "joint_record": joint,
            "outcome": 0,
        })

    out_path = tmp_path / "serpo.parquet"
    n_written = build_dataset_for_task_group(
        inputs, tokenizer=tok, method="serpo", out_path=out_path
    )
    assert n_written == 8

    # Parquet sanity
    table = pq.read_table(out_path)
    df = table.to_pandas()
    assert len(df) == 8
    assert {"task_id", "seed", "input_ids", "response_mask",
            "advantages", "outcome"}.issubset(df.columns)
    # First row sanity
    row0 = df.iloc[0]
    assert len(row0["input_ids"]) == len(row0["advantages"])
    assert len(row0["input_ids"]) == len(row0["response_mask"])


def test_build_dataset_vanilla_failonly_all_zero_advantage(tmp_path):
    """In fail-only set with outcome=0 across 8 seeds, vanilla advantage is all zero."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    joint = json.loads(FIXTURE_JOINT.read_text())
    inputs = [{
        "task_id": "07b42fd_1", "seed": s,
        "lm_calls_path": FIXTURE_LM_CALLS, "joint_record": joint, "outcome": 0,
    } for s in range(1, 9)]

    out_path = tmp_path / "vanilla.parquet"
    build_dataset_for_task_group(inputs, tokenizer=tok, method="vanilla", out_path=out_path)
    df = pq.read_table(out_path).to_pandas()
    for _, row in df.iterrows():
        adv = row["advantages"]
        # All zeros (because sigma == eps and numerator == 0)
        assert all(a == 0.0 for a in adv)
```

- [ ] **Step 4.3: Run test, expect ImportError**

```bash
cd /data/minjeong/Autonomous_agent && python -m pytest grpo/tests/test_build_dataset.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4.4: Implement `grpo/preprocess/build_dataset.py`**

```python
"""Build per-method parquet datasets from Round-0 rollouts + joint reward output.

CLI:
  python -m grpo.preprocess.build_dataset \
      --method {vanilla,serpo} \
      --condition failonly \
      --rollout-dir appworld/experiments/outputs/rollout/round0 \
      --joint-dir rubric_reward/results/rollout \
      --output-dir grpo/data/round0
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from grpo.preprocess.failset import build_failset
from grpo.preprocess.tokenize_trajectory import tokenize_trajectory
from grpo.preprocess.make_advantages import (
    RolloutData, compute_vanilla_advantage, compute_serpo_advantage,
)

logger = logging.getLogger(__name__)

MAX_TOKENS = 32000


def _load_outcomes(rollout_dir: Path, seed: int) -> dict[str, bool]:
    eval_path = rollout_dir / f"seed_{seed}" / "evaluations" / "train.json"
    data = json.loads(eval_path.read_text())
    return {tid: rec.get("success", False) for tid, rec in data["individual"].items()}


def _load_joint_records(joint_dir: Path, seed: int) -> dict[str, dict[str, Any]]:
    """Load joint_seed_N.jsonl into {task_id: record}."""
    path = joint_dir / f"joint_seed_{seed}.jsonl"
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        logger.warning("joint file missing: %s", path)
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["task_id"]] = rec
    return out


def build_dataset_for_task_group(
    inputs: list[dict[str, Any]],
    *,
    tokenizer,
    method: str,
    out_path: Path | None = None,
) -> int:
    """Process 8 rollouts for one task (group), compute method's advantage, append to parquet.

    Each input dict: {task_id, seed, lm_calls_path, joint_record, outcome}
    Returns number of rollouts written (0 if any in the group fails filtering — drop whole group).
    """
    rollouts: list[RolloutData] = []
    for inp in inputs:
        try:
            tok_result = tokenize_trajectory(inp["lm_calls_path"], tokenizer)
        except Exception as e:
            logger.warning("tokenize failed for task=%s seed=%s: %s",
                           inp["task_id"], inp["seed"], e)
            return 0
        if len(tok_result["input_ids"]) > MAX_TOKENS:
            logger.warning("len > %d for task=%s seed=%s — dropping whole group",
                           MAX_TOKENS, inp["task_id"], inp["seed"])
            return 0
        joint = inp["joint_record"]
        rollouts.append(RolloutData(
            task_id=inp["task_id"], seed=inp["seed"],
            input_ids=tok_result["input_ids"],
            attention_mask=tok_result["attention_mask"],
            response_mask=tok_result["response_mask"],
            step_token_ranges=tok_result["step_token_ranges"],
            segments=joint.get("segments", []),
            outcome=int(inp["outcome"]),
        ))

    if method == "vanilla":
        compute_vanilla_advantage(rollouts)
    elif method == "serpo":
        compute_serpo_advantage(rollouts)
    else:
        raise ValueError(f"unknown method: {method}")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist([
            {
                "task_id": r.task_id, "seed": r.seed,
                "input_ids": r.input_ids,
                "attention_mask": r.attention_mask,
                "response_mask": r.response_mask,
                "advantages": [float(x) for x in r.token_adv.tolist()],
                "outcome": r.outcome,
                "num_steps": len(r.step_token_ranges),
            }
            for r in rollouts
        ])
        if out_path.exists():
            existing = pq.read_table(out_path)
            table = pa.concat_tables([existing, table])
        pq.write_table(table, out_path)

    return len(rollouts)


def run_build(method: str, condition: str, rollout_dir: Path, joint_dir: Path,
              output_dir: Path, tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if condition != "failonly":
        raise NotImplementedError("v1 supports only condition=failonly")

    failset = set(build_failset(rollout_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "failonly_task_ids.json").write_text(
        json.dumps(sorted(failset), indent=2)
    )
    logger.info("fail-only task count: %d", len(failset))

    # Pre-load all outcomes + joint records per seed
    seed_outcomes: dict[int, dict[str, bool]] = {}
    seed_joints: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in range(1, 9):
        seed_outcomes[seed] = _load_outcomes(rollout_dir, seed)
        seed_joints[seed] = _load_joint_records(joint_dir, seed)

    out_path = output_dir / f"{method}_failonly.parquet"
    if out_path.exists():
        out_path.unlink()  # overwrite

    n_groups = 0
    n_rows = 0
    n_skipped_groups = 0
    skipped_log_path = output_dir / "skipped.jsonl"
    with skipped_log_path.open("w") as skip_log:
        for task_id in sorted(failset):
            inputs: list[dict[str, Any]] = []
            group_ok = True
            for seed in range(1, 9):
                lm_calls = (
                    rollout_dir / f"seed_{seed}" / "tasks" / task_id
                    / "logs" / "lm_calls.jsonl"
                )
                if not lm_calls.exists():
                    skip_log.write(json.dumps({"task_id": task_id, "seed": seed,
                                               "reason": "lm_calls_missing"}) + "\n")
                    group_ok = False
                    break
                joint_rec = seed_joints[seed].get(task_id)
                if joint_rec is None and method == "serpo":
                    skip_log.write(json.dumps({"task_id": task_id, "seed": seed,
                                               "reason": "joint_missing"}) + "\n")
                    group_ok = False
                    break
                # Vanilla doesn't need joint segments but we still record it (may be None)
                inputs.append({
                    "task_id": task_id, "seed": seed,
                    "lm_calls_path": lm_calls,
                    "joint_record": joint_rec or {"segments": []},
                    "outcome": int(seed_outcomes[seed].get(task_id, False)),
                })
            if not group_ok:
                n_skipped_groups += 1
                continue

            written = build_dataset_for_task_group(
                inputs, tokenizer=tokenizer, method=method, out_path=out_path,
            )
            if written == 0:
                n_skipped_groups += 1
                skip_log.write(json.dumps({"task_id": task_id,
                                           "reason": "tokenize_or_length"}) + "\n")
            else:
                n_groups += 1
                n_rows += written

    logger.info("done: %d groups written (%d rows), %d groups skipped — parquet: %s",
                n_groups, n_rows, n_skipped_groups, out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["vanilla", "serpo"], required=True)
    ap.add_argument("--condition", choices=["failonly"], default="failonly")
    ap.add_argument("--rollout-dir", type=Path,
                    default=Path("appworld/experiments/outputs/rollout/round0"))
    ap.add_argument("--joint-dir", type=Path,
                    default=Path("rubric_reward/results/rollout"))
    ap.add_argument("--output-dir", type=Path, default=Path("grpo/data/round0"))
    args = ap.parse_args()
    run_build(args.method, args.condition, args.rollout_dir, args.joint_dir, args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.5: Run tests, expect PASS**

```bash
cd /data/minjeong/Autonomous_agent && python -m pytest grpo/tests/test_build_dataset.py -v
```
Expected: 2 passed.

- [ ] **Step 4.6: Commit**

```bash
git add grpo/preprocess/build_dataset.py grpo/tests/test_build_dataset.py grpo/tests/fixtures/joint_07b42fd_1.json && git commit -m "feat(preprocess): orchestrator producing per-method parquet datasets"
```

---

## Task 5: verl-based GRPO trainer (offline)

**Files:**
- Create: `grpo/trainer/dataset.py`, `grpo/trainer/offline_trainer.py`, `grpo/trainer/run_grpo.py`
- Create configs: `grpo/trainer/config/{base_round1,vanilla_round1,serpo_round1}.yaml`

This is the most complex task. Strategy: minimize verl coupling by using verl utilities (`compute_kl`, `compute_policy_loss`, FSDP setup) inside a thin custom training loop built directly on `accelerate` + `peft`. Skip verl's full Ray-based RayPPOTrainer.

- [ ] **Step 5.1: Inspect verl utility imports we need**

```bash
python -c "from verl.utils.torch_functional import logprobs_from_logits; print('ok')"
python -c "from verl import DataProto; print('ok')"
python -c "from verl.utils.kl_estimator import compute_kl_k3 if hasattr(__import__('verl.utils.kl_estimator', fromlist=['compute_kl_k3']),'compute_kl_k3') else None; print('check k3')"
```
Expected: `ok` for first two. K3 may live at `verl.utils.torch_functional.kl_penalty` or similar — adjust import in step 5.4 based on what's available.

- [ ] **Step 5.2: Implement `grpo/trainer/dataset.py`**

```python
"""ParquetGRPODataset: yields one trajectory per item with precomputed advantages."""
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq


class ParquetGRPODataset(Dataset):
    def __init__(self, parquet_path: Path):
        self.table = pq.read_table(parquet_path)
        self.df = self.table.to_pandas()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(row["attention_mask"], dtype=torch.long)
        response_mask = torch.tensor(row["response_mask"], dtype=torch.long)
        advantages = torch.tensor(row["advantages"], dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "advantages": advantages,
            "task_id": row["task_id"],
            "seed": int(row["seed"]),
        }


def collate_pad_right(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch of variable-length sequences."""
    T = max(item["input_ids"].size(0) for item in batch)
    B = len(batch)

    def pad(t: torch.Tensor, val) -> torch.Tensor:
        pad_len = T - t.size(0)
        if pad_len == 0:
            return t
        return torch.cat([t, torch.full((pad_len,), val, dtype=t.dtype)])

    return {
        "input_ids": torch.stack([pad(b["input_ids"], pad_token_id) for b in batch]),
        "attention_mask": torch.stack([pad(b["attention_mask"], 0) for b in batch]),
        "response_mask": torch.stack([pad(b["response_mask"], 0) for b in batch]),
        "advantages": torch.stack([pad(b["advantages"], 0.0) for b in batch]),
        "task_ids": [b["task_id"] for b in batch],
        "seeds": [b["seed"] for b in batch],
    }
```

- [ ] **Step 5.3: Implement `grpo/trainer/offline_trainer.py` (the core loop)**

```python
"""Offline GRPO trainer.

Loads a parquet dataset of trajectories with precomputed per-token advantages,
runs K optimizer steps of clipped GRPO with k3 KL-to-ref using peft.disable_adapter.
"""
from __future__ import annotations
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from grpo.trainer.dataset import ParquetGRPODataset, collate_pad_right

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    parquet_path: str = ""
    output_dir: str = ""
    K_steps: int = 20
    mini_batch_size: int = 8
    micro_batch_size: int = 1
    lr: float = 1e-6
    clip_eps: float = 0.2
    kl_beta: float = 0.01
    grad_clip: float = 1.0
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    seed: int = 42
    log_every: int = 1


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits: (B, T, V); labels: (B, T) -> per-token logprob (B, T)."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def shift_for_causal_lm(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       response_mask: torch.Tensor, advantages: torch.Tensor):
    """Shift labels/masks/advantages for next-token prediction. Returns:
      input_ids[:, :-1], labels=input_ids[:, 1:], shifted masks & advantages."""
    return (
        input_ids[:, :-1],
        input_ids[:, 1:],
        attention_mask[:, 1:],
        response_mask[:, 1:],
        advantages[:, 1:],
    )


def compute_logprobs(model, tokenizer, batch: dict[str, torch.Tensor],
                     micro_bsz: int = 1) -> torch.Tensor:
    """Forward pass returning per-token logprobs (B, T-1)."""
    inputs, labels, attn, _, _ = shift_for_causal_lm(
        batch["input_ids"], batch["attention_mask"],
        batch["response_mask"], batch["advantages"],
    )
    B = inputs.size(0)
    all_lp = []
    for s in range(0, B, micro_bsz):
        sl = slice(s, s + micro_bsz)
        out = model(input_ids=inputs[sl].to(model.device),
                    attention_mask=attn[sl].to(model.device))
        lp = logprobs_from_logits(out.logits, labels[sl].to(model.device))
        all_lp.append(lp.detach().cpu())
    return torch.cat(all_lp, dim=0)


def run_training(cfg: GRPOConfig) -> None:
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading tokenizer + base model (%s)", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_target_modules),
        lora_dropout=cfg.lora_dropout, bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    dataset = ParquetGRPODataset(Path(cfg.parquet_path))
    logger.info("dataset rows: %d", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=cfg.mini_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pad_right(b, tokenizer.pad_token_id),
        drop_last=True,
    )

    # Pre-compute old_logp and ref_logp at round start.
    # old_logp = logp under LoRA-at-round-start (Round 1: LoRA is freshly init,
    #            but its zero-init delta makes old == ref initially. We still
    #            compute via the LoRA-enabled forward for correctness in future rounds.)
    # ref_logp = logp with adapter disabled (= base model only).
    model.eval()
    cached: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for batch in loader:
            # old: LoRA enabled
            old_lp = compute_logprobs(model, tokenizer, batch, micro_bsz=cfg.micro_batch_size)
            # ref: LoRA disabled
            with model.disable_adapter():
                ref_lp = compute_logprobs(model, tokenizer, batch, micro_bsz=cfg.micro_batch_size)
            cached.append({
                **batch,
                "old_logp": old_lp,
                "ref_logp": ref_lp,
            })
    logger.info("cached old/ref logp for %d mini-batches", len(cached))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
    )
    model.train()

    metrics_log = []
    step_iter = 0
    while step_iter < cfg.K_steps:
        for batch in cached:
            if step_iter >= cfg.K_steps:
                break
            input_ids = batch["input_ids"]
            attn = batch["attention_mask"]
            resp_mask = batch["response_mask"]
            adv = batch["advantages"]
            old_lp = batch["old_logp"]
            ref_lp = batch["ref_logp"]

            inputs, labels, attn_s, resp_s, adv_s = shift_for_causal_lm(
                input_ids, attn, resp_mask, adv
            )
            B = inputs.size(0)
            optimizer.zero_grad()
            total_loss = 0.0
            total_pg = 0.0
            total_kl = 0.0
            total_resp = 0.0
            for s in range(0, B, cfg.micro_batch_size):
                sl = slice(s, s + cfg.micro_batch_size)
                out = model(
                    input_ids=inputs[sl].to(model.device),
                    attention_mask=attn_s[sl].to(model.device),
                )
                new_lp = logprobs_from_logits(out.logits, labels[sl].to(model.device))
                old_lp_d = old_lp[sl].to(model.device)
                ref_lp_d = ref_lp[sl].to(model.device)
                resp_d = resp_s[sl].to(model.device).float()
                adv_d = adv_s[sl].to(model.device)

                # ratio = exp(new - old); clip
                log_ratio = new_lp - old_lp_d
                ratio = log_ratio.exp()
                ratio_clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
                pg_unclipped = ratio * adv_d
                pg_clipped = ratio_clipped * adv_d
                pg = -torch.min(pg_unclipped, pg_clipped)
                pg_loss = (pg * resp_d).sum() / (resp_d.sum() + 1e-8)

                # k3 KL: exp(r - n) - (r - n) - 1
                ref_minus_new = ref_lp_d - new_lp
                kl_per_tok = ref_minus_new.exp() - ref_minus_new - 1.0
                kl_loss = (kl_per_tok * resp_d).sum() / (resp_d.sum() + 1e-8)

                loss = pg_loss + cfg.kl_beta * kl_loss
                # micro-batch loss scaled for gradient accumulation
                scale = cfg.micro_batch_size / max(B, 1)
                (loss * scale).backward()

                bs_count = max(int(resp_d.sum().item()), 1)
                total_loss += loss.item() * bs_count
                total_pg += pg_loss.item() * bs_count
                total_kl += kl_loss.item() * bs_count
                total_resp += bs_count

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip,
            )
            optimizer.step()

            avg_loss = total_loss / max(total_resp, 1)
            avg_pg = total_pg / max(total_resp, 1)
            avg_kl = total_kl / max(total_resp, 1)
            metrics_log.append({
                "step": step_iter, "loss": avg_loss, "pg_loss": avg_pg,
                "kl_loss": avg_kl, "resp_tokens": int(total_resp),
            })
            if step_iter % cfg.log_every == 0:
                logger.info(
                    "step %d  loss=%.4f  pg=%.4f  kl=%.4f  resp_toks=%d",
                    step_iter, avg_loss, avg_pg, avg_kl, total_resp,
                )
            step_iter += 1

    # Save LoRA adapter
    model.save_pretrained(out_dir)
    (out_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(m) for m in metrics_log) + "\n"
    )
    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))
    logger.info("saved LoRA to %s", out_dir)
```

- [ ] **Step 5.4: Implement `grpo/trainer/run_grpo.py` (Hydra-free CLI for v1)**

```python
"""Entry point for GRPO training. Reads a YAML config and runs offline_trainer."""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import yaml

from grpo.trainer.offline_trainer import GRPOConfig, run_training


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg_dict = yaml.safe_load(args.config.read_text())
    cfg = GRPOConfig(**cfg_dict)
    run_training(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.5: Write configs**

`grpo/trainer/config/vanilla_round1.yaml`:
```yaml
model_name: Qwen/Qwen2.5-7B-Instruct
parquet_path: grpo/data/round0/vanilla_failonly.parquet
output_dir: grpo/ckpts/vanilla/round1
K_steps: 20
mini_batch_size: 8
micro_batch_size: 1
lr: 1e-6
clip_eps: 0.2
kl_beta: 0.01
grad_clip: 1.0
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.0
seed: 42
```

`grpo/trainer/config/serpo_round1.yaml`:
```yaml
model_name: Qwen/Qwen2.5-7B-Instruct
parquet_path: grpo/data/round0/serpo_failonly.parquet
output_dir: grpo/ckpts/serpo/round1
K_steps: 20
mini_batch_size: 8
micro_batch_size: 1
lr: 1e-6
clip_eps: 0.2
kl_beta: 0.01
grad_clip: 1.0
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.0
seed: 42
```

- [ ] **Step 5.6: Smoke test — K=2 steps on fixture**

Create `grpo/tests/test_trainer_smoke.py`:

```python
"""Smoke test for the trainer: 2 grad steps on a tiny fixture parquet."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

from grpo.trainer.offline_trainer import GRPOConfig, run_training


@pytest.mark.slow
def test_trainer_runs_two_steps(tmp_path):
    """Build a 16-row tiny parquet and verify 2 grad steps don't NaN."""
    # 16 rows × short sequences
    rng = np.random.default_rng(0)
    rows = []
    for i in range(16):
        T = 32 + rng.integers(0, 16)
        input_ids = rng.integers(0, 1000, size=T).tolist()
        resp = [0] * 8 + [1] * (T - 16) + [0] * 8  # middle is "response"
        adv = [(0.1 if m else 0.0) for m in resp]
        rows.append({
            "task_id": f"t{i}", "seed": i,
            "input_ids": input_ids,
            "attention_mask": [1] * T,
            "response_mask": resp,
            "advantages": adv,
            "outcome": 0, "num_steps": 1,
        })
    parquet_path = tmp_path / "tiny.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)

    cfg = GRPOConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        parquet_path=str(parquet_path),
        output_dir=str(tmp_path / "ckpt"),
        K_steps=2,
        mini_batch_size=8, micro_batch_size=1,
        lr=1e-6, kl_beta=0.01, clip_eps=0.2,
    )
    run_training(cfg)

    metrics = [json.loads(l) for l in (tmp_path / "ckpt" / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 2
    for m in metrics:
        assert m["loss"] == m["loss"]  # not NaN
        assert m["kl_loss"] >= -1e-3
```

```bash
cd /data/minjeong/Autonomous_agent && python -m pytest grpo/tests/test_trainer_smoke.py -v -m slow
```
Expected: 1 passed. Note: requires GPU memory for the 7B base model. If OOM, downsize to a smaller test model like `Qwen/Qwen2.5-0.5B-Instruct` for this smoke test only.

- [ ] **Step 5.7: Commit**

```bash
git add grpo/trainer/ grpo/tests/test_trainer_smoke.py && git commit -m "feat(trainer): offline GRPO loop with peft disable_adapter + k3 KL"
```

---

## Task 6: `eval/run_test_normal.py` — test_normal eval via vLLM LoRA hot-swap

**Files:**
- Create: `grpo/eval/run_test_normal.py`

This task requires the external vLLM to be running (started via `serve_qwen_vllm.sh` with `--enable-lora`). We push the LoRA adapter to vLLM's runtime LoRA registry, then invoke `appworld run` against test_normal pointing at the LoRA-named model.

- [ ] **Step 6.1: Implement `grpo/eval/run_test_normal.py`**

```python
"""Evaluate a trained LoRA on AppWorld test_normal.

Steps:
  1. Register LoRA with the running vLLM (POST /v1/load_lora_adapter).
  2. Invoke appworld run test_normal pointing at the LoRA-named model.
  3. Parse evaluations/test_normal.json → success rate.
  4. Write metrics file.
"""
from __future__ import annotations
import argparse
import json
import logging
import subprocess
from pathlib import Path
import requests

logger = logging.getLogger(__name__)


def register_lora(vllm_url: str, lora_name: str, lora_path: Path) -> None:
    payload = {"lora_name": lora_name, "lora_path": str(lora_path)}
    r = requests.post(f"{vllm_url}/v1/load_lora_adapter", json=payload, timeout=120)
    r.raise_for_status()
    logger.info("registered LoRA %s ← %s", lora_name, lora_path)


def unregister_lora(vllm_url: str, lora_name: str) -> None:
    try:
        r = requests.post(
            f"{vllm_url}/v1/unload_lora_adapter",
            json={"lora_name": lora_name},
            timeout=60,
        )
        r.raise_for_status()
    except Exception as e:
        logger.warning("unregister LoRA %s failed: %s", lora_name, e)


def run_appworld_test(experiment_name: str, agent_name: str, num_processes: int,
                      appworld_root: Path) -> Path:
    """Run appworld test_normal; return path to evaluations/test_normal.json."""
    cmd = [
        "appworld", "run", experiment_name,
        "--agent-name", agent_name,
        "--num-processes", str(num_processes),
        "--with-evaluation",
    ]
    logger.info("running: %s (cwd=%s)", " ".join(cmd), appworld_root)
    subprocess.run(cmd, cwd=appworld_root, check=True)
    return appworld_root / "experiments" / "outputs" / experiment_name / "evaluations" / "test_normal.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--method", choices=["vanilla", "serpo"], required=True)
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--appworld-root", type=Path,
                    default=Path("/data/minjeong/Autonomous_agent/appworld"))
    ap.add_argument("--agent-name", default="simplified_react_code_agent")
    ap.add_argument("--num-processes", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    lora_name = f"qwen25_7b_lora_{args.method}_round1"
    experiment_name = f"eval/test_normal/{lora_name}"

    register_lora(args.vllm_url, lora_name, args.ckpt)
    try:
        eval_path = run_appworld_test(
            experiment_name, args.agent_name, args.num_processes, args.appworld_root,
        )
        data = json.loads(eval_path.read_text())
        success_rate = data["aggregate"]["task_goal_completion"]
        per_task = {tid: rec.get("success", False) for tid, rec in data["individual"].items()}
        out = {
            "method": args.method, "lora_name": lora_name,
            "success_rate": success_rate,
            "n_tasks": len(per_task),
            "n_success": sum(per_task.values()),
            "per_task": per_task,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2))
        logger.info("wrote %s — success_rate=%.2f%%", args.output, success_rate)
    finally:
        unregister_lora(args.vllm_url, lora_name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Manual sanity check (after a real LoRA is trained)**

```bash
# requires vLLM running with --enable-lora
python -m grpo.eval.run_test_normal \
    --ckpt grpo/ckpts/vanilla/round1 \
    --method vanilla \
    --output grpo/metrics/vanilla_round1.json
```

(Don't expect this to pass until Task 5's training has produced a real LoRA. Manual test only.)

- [ ] **Step 6.3: Commit**

```bash
git add grpo/eval/ && git commit -m "feat(eval): test_normal evaluation via vLLM LoRA hot-swap"
```

---

## Task 7: Orchestration shell scripts

**Files:**
- Create: `grpo/scripts/train_vanilla_round1.sh`, `grpo/scripts/train_serpo_round1.sh`

- [ ] **Step 7.1: Write `grpo/scripts/train_vanilla_round1.sh`**

```bash
#!/usr/bin/env bash
# End-to-end Vanilla GRPO Round-1 run: preprocess → train → eval.
set -euo pipefail

cd /data/minjeong/Autonomous_agent

echo "[1/3] build vanilla parquet"
python -m grpo.preprocess.build_dataset \
    --method vanilla --condition failonly \
    --rollout-dir appworld/experiments/outputs/rollout/round0 \
    --joint-dir rubric_reward/results/rollout \
    --output-dir grpo/data/round0

echo "[2/3] train"
python -m grpo.trainer.run_grpo --config grpo/trainer/config/vanilla_round1.yaml

echo "[3/3] eval test_normal (requires vLLM running with --enable-lora)"
python -m grpo.eval.run_test_normal \
    --ckpt grpo/ckpts/vanilla/round1 \
    --method vanilla \
    --output grpo/metrics/vanilla_round1.json

echo "done. metrics: grpo/metrics/vanilla_round1.json"
```

- [ ] **Step 7.2: Write `grpo/scripts/train_serpo_round1.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /data/minjeong/Autonomous_agent

echo "[1/3] build serpo parquet"
python -m grpo.preprocess.build_dataset \
    --method serpo --condition failonly \
    --rollout-dir appworld/experiments/outputs/rollout/round0 \
    --joint-dir rubric_reward/results/rollout \
    --output-dir grpo/data/round0

echo "[2/3] train"
python -m grpo.trainer.run_grpo --config grpo/trainer/config/serpo_round1.yaml

echo "[3/3] eval test_normal"
python -m grpo.eval.run_test_normal \
    --ckpt grpo/ckpts/serpo/round1 \
    --method serpo \
    --output grpo/metrics/serpo_round1.json

echo "done. metrics: grpo/metrics/serpo_round1.json"
```

- [ ] **Step 7.3: chmod +x**

```bash
chmod +x grpo/scripts/*.sh
```

- [ ] **Step 7.4: Commit**

```bash
git add grpo/scripts/ && git commit -m "feat(scripts): vanilla and serpo round-1 orchestration"
```

---

## Final acceptance criteria

- All unit tests pass: `pytest grpo/tests/ -v` (excluding `@pytest.mark.slow` smoke if no GPU available)
- `grpo/data/round0/{vanilla,serpo}_failonly.parquet` build without error (after Phase 2 joint output exists)
- `grpo/scripts/train_vanilla_round1.sh` runs end-to-end → LoRA checkpoint produced + metrics file
- `grpo/scripts/train_serpo_round1.sh` runs end-to-end → LoRA checkpoint produced + metrics file
- Both metrics files contain `success_rate` and `per_task` keys

## Open execution risks

- **verl install** may have heavy transitive deps (Ray, vllm-as-dep). If install hangs or conflicts, the trainer code in Task 5 does NOT import verl directly; we use peft + torch directly. The `pip install verl` step can be removed from Step 0.2 if it blocks (note in commit). We keep verl listed in pyproject.toml as a soft optional for future iteration.
- **32k context + 7B + LoRA + gradient checkpointing** may still OOM on one 96GB GPU during training. Mitigation: lower `mini_batch_size` to 4, or shard with FSDP (accelerate config). Adjust in `grpo/trainer/config/*.yaml`.
- **vLLM LoRA hot-swap API endpoint** (`/v1/load_lora_adapter`) availability depends on the vLLM version. If not exposed, fall back to relaunching vLLM with `--lora-modules` flag pointing at the checkpoint.
