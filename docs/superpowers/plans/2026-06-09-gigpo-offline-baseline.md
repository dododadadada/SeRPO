# GiGPO Offline Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GiGPO (Group-in-Group Policy Optimization, arXiv 2505.10978) as a fourth advantage method in the offline GRPO preprocessing pipeline, so it can serve as the outcome-reward / step-granularity baseline against SeRPO.

**Architecture:** GiGPO is a new advantage function `compute_gigpo_advantage` in `grpo/preprocess/make_advantages.py`, parallel to the existing `vanilla`/`serpo`/`serpo_avg`. It needs a per-step "anchor observation" (the env-output text preceding each assistant step), which `tokenize_trajectory` will emit as a new additive field. `build_dataset.py` gets a `gigpo` method branch + CLI args. The trainer, dataset loader, and other advantage functions are untouched. Faithful to the paper's TeX: `A = A^E + ω·A^S`, γ=0.95, ω=1, F_norm=1 (leave-one-out) default, similarity threshold 0.9.

**Tech Stack:** Python, numpy, pyarrow, transformers tokenizer; pytest. Repo conventions: `from __future__ import annotations`, dataclasses, type hints, module-level logger.

**Spec:** `docs/superpowers/specs/2026-06-09-gigpo-offline-baseline-design.md`

---

## File Structure

- **Modify** `grpo/preprocess/make_advantages.py` — add `step_anchor_obs` field to `RolloutData`; add `compute_gigpo_advantage` + helpers (`_to_hashable`, `_are_similar`, `_build_step_groups`, `_discounted_step_returns`, `_loo_norm`).
- **Modify** `grpo/preprocess/tokenize_trajectory.py` — return `step_anchor_obs: list[str]` (one anchor string per step) additively.
- **Modify** `grpo/preprocess/build_dataset.py` — add `"gigpo"` method, dispatch branch, CLI args, fail-only guard, anchor-obs wiring.
- **Modify** `grpo/tests/test_make_advantages.py` — GiGPO unit tests.
- **Modify** `grpo/tests/test_tokenize_trajectory.py` — assert `step_anchor_obs` shape/content.
- **Create** `grpo/trainer/config/gigpo_9b_full_binary_v3i.yaml` — training config (clone of vanilla full-binary).
- **Create** `grpo/scripts/train_gigpo_9b_full_binary_v3i.sh` — training launch script.

---

## Task 1: Add `step_anchor_obs` field to `RolloutData`

**Files:**
- Modify: `grpo/preprocess/make_advantages.py` (the `RolloutData` dataclass, lines 16-32)
- Test: `grpo/tests/test_make_advantages.py`

- [ ] **Step 1: Write the failing test**

Add to `grpo/tests/test_make_advantages.py`:

```python
def test_rollout_data_has_step_anchor_obs_default():
    """RolloutData accepts an optional step_anchor_obs, defaulting to empty list."""
    r = RolloutData(
        task_id="t", seed=1,
        input_ids=[0, 1, 2], attention_mask=[1, 1, 1],
        response_mask=[0, 1, 1], step_token_ranges=[(1, 3)],
        segments=[], outcome=0.0,
    )
    assert r.step_anchor_obs == []
    r2 = RolloutData(
        task_id="t", seed=1,
        input_ids=[0, 1, 2], attention_mask=[1, 1, 1],
        response_mask=[0, 1, 1], step_token_ranges=[(1, 3)],
        segments=[], outcome=0.0, step_anchor_obs=["obs1"],
    )
    assert r2.step_anchor_obs == ["obs1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grpo/tests/test_make_advantages.py::test_rollout_data_has_step_anchor_obs_default -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'step_anchor_obs'`

- [ ] **Step 3: Add the field**

In `grpo/preprocess/make_advantages.py`, in the `RolloutData` dataclass, add this field immediately after the `outcome: float` line and before `token_adv`:

```python
    # Per-step anchor observation (env-output text preceding each assistant
    # step). Used only by GiGPO; empty for other methods. Length == num_steps.
    step_anchor_obs: list[str] = field(default_factory=list)
```

(`field` is already imported at the top: `from dataclasses import dataclass, field`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest grpo/tests/test_make_advantages.py::test_rollout_data_has_step_anchor_obs_default -v`
Expected: PASS

- [ ] **Step 5: Run the full existing suite to confirm no regression**

Run: `python -m pytest grpo/tests/test_make_advantages.py -v`
Expected: all PASS (existing tests construct RolloutData without the new field — the default makes them still valid)

- [ ] **Step 6: Commit**

```bash
git add grpo/preprocess/make_advantages.py grpo/tests/test_make_advantages.py
git commit -m "feat(preprocess): add step_anchor_obs field to RolloutData for GiGPO"
```

---

## Task 2: `_loo_norm` and `_to_hashable` / `_are_similar` helpers

**Files:**
- Modify: `grpo/preprocess/make_advantages.py`
- Test: `grpo/tests/test_make_advantages.py`

- [ ] **Step 1: Write the failing tests**

Add to `grpo/tests/test_make_advantages.py` (and extend its imports — change the import block at top to include the new names):

```python
from grpo.preprocess.make_advantages import (
    RolloutData,
    compute_serpo_advantage,
    compute_vanilla_advantage,
    _loo_norm,
    _to_hashable,
    _are_similar,
)
```

```python
def test_loo_norm_subtracts_mean_only():
    """leave_one_out mode subtracts the group mean, no division by std."""
    vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = _loo_norm(vals, mode="leave_one_out")
    assert np.allclose(out, [-1.0, 0.0, 1.0])  # mean=2


def test_loo_norm_std_mode_divides():
    """std mode divides by population std."""
    vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = _loo_norm(vals, mode="std")
    expected = (vals - 2.0) / (np.std(vals) + 1e-8)
    assert np.allclose(out, expected)


def test_loo_norm_singleton_is_zero():
    """A single-element group has zero advantage in both modes (no peers)."""
    assert np.allclose(_loo_norm(np.array([5.0], dtype=np.float32),
                                 mode="leave_one_out"), [0.0])
    assert np.allclose(_loo_norm(np.array([5.0], dtype=np.float32),
                                 mode="std"), [0.0])


def test_to_hashable_strings_and_lists():
    assert _to_hashable("abc") == "abc"
    assert _to_hashable(["a", "b"]) == ("a", "b")


def test_are_similar_threshold():
    assert _are_similar("Output: ok", "Output: ok", 0.9) is True
    assert _are_similar("Output: ok", "totally different text here", 0.9) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest grpo/tests/test_make_advantages.py -k "loo_norm or to_hashable or are_similar" -v`
Expected: FAIL with `ImportError: cannot import name '_loo_norm'`

- [ ] **Step 3: Implement the helpers**

In `grpo/preprocess/make_advantages.py`, add after the `EPS = 1e-8` line and before `RolloutData` (so they're available throughout):

```python
from difflib import SequenceMatcher


def _to_hashable(x):
    """Convert an observation into a hashable key for anchor-state grouping.
    Ported from verl-agent gigpo/core_gigpo.py (Apache-2.0)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return tuple(x.flatten().tolist())
    if isinstance(x, (list, tuple)):
        return tuple(_to_hashable(e) for e in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _to_hashable(v)) for k, v in x.items()))
    raise TypeError(f"unhashable observation type: {type(x)}")


def _are_similar(a: str, b: str, threshold: float) -> bool:
    """True if longest-matching-subsequence ratio >= threshold.
    Ported from verl-agent gigpo/core_gigpo.py (Apache-2.0)."""
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("similarity-based grouping supports only str observations")
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _loo_norm(values: np.ndarray, *, mode: str = "leave_one_out") -> np.ndarray:
    """Normalize a group of scalars by subtracting the group mean.

    mode='leave_one_out' (paper F_norm=1): subtract mean only (a rescaled RLOO;
      the rescale is absorbed into the learning rate).
    mode='std' (paper F_norm=std): also divide by population std.
    A singleton group (len 1) returns 0 (no peers to compare against), matching
    the GiGPO reference's size-1 handling.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size <= 1:
        return np.zeros_like(values)
    centered = values - float(values.mean())
    if mode == "leave_one_out":
        return centered
    if mode == "std":
        return centered / (float(values.std()) + EPS)
    raise ValueError(f"unknown norm mode: {mode!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest grpo/tests/test_make_advantages.py -k "loo_norm or to_hashable or are_similar" -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add grpo/preprocess/make_advantages.py grpo/tests/test_make_advantages.py
git commit -m "feat(preprocess): add GiGPO helpers (loo_norm, to_hashable, are_similar)"
```

---

## Task 3: `_discounted_step_returns` and `_build_step_groups`

**Files:**
- Modify: `grpo/preprocess/make_advantages.py`
- Test: `grpo/tests/test_make_advantages.py`

- [ ] **Step 1: Write the failing tests**

Add the two new names to the import block in `grpo/tests/test_make_advantages.py`:

```python
from grpo.preprocess.make_advantages import (
    RolloutData,
    compute_serpo_advantage,
    compute_vanilla_advantage,
    _loo_norm,
    _to_hashable,
    _are_similar,
    _discounted_step_returns,
    _build_step_groups,
)
```

```python
def test_discounted_step_returns_terminal_only():
    """R_k = gamma^(N-k) * outcome for terminal-only reward, k=1..N (1-indexed)."""
    # 3 steps, outcome 1.0, gamma 0.95 -> [0.95^2, 0.95^1, 0.95^0]
    out = _discounted_step_returns(num_steps=3, outcome=1.0, gamma=0.95)
    assert np.allclose(out, [0.95 ** 2, 0.95, 1.0])
    # outcome 0 -> all zeros
    assert np.allclose(_discounted_step_returns(3, 0.0, 0.95), [0.0, 0.0, 0.0])


def test_build_step_groups_exact_match():
    """Steps with identical anchor strings get the same group id."""
    # 2 rollouts, 2 steps each. Anchors: r0=[A,B], r1=[A,C]
    anchors = [["A", "B"], ["A", "C"]]
    groups = _build_step_groups(anchors, enable_similarity=False, threshold=0.9)
    # groups[i] is a list parallel to anchors[i]; the two "A" steps share an id
    assert groups[0][0] == groups[1][0]      # both "A"
    assert groups[0][1] != groups[0][0]      # "B" different from "A"
    assert groups[0][1] != groups[1][1]      # "B" != "C"


def test_build_step_groups_similarity():
    """Near-identical anchors cluster together when similarity is enabled."""
    anchors = [["Output: ok aaaa"], ["Output: ok aaab"]]
    groups = _build_step_groups(anchors, enable_similarity=True, threshold=0.9)
    assert groups[0][0] == groups[1][0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest grpo/tests/test_make_advantages.py -k "discounted_step_returns or build_step_groups" -v`
Expected: FAIL with `ImportError: cannot import name '_discounted_step_returns'`

- [ ] **Step 3: Implement**

In `grpo/preprocess/make_advantages.py`, add after `_loo_norm`:

```python
def _discounted_step_returns(
    num_steps: int, outcome: float, gamma: float,
) -> np.ndarray:
    """Terminal-only discounted return-to-go per step (paper Eq. 5 specialized
    to a single terminal reward). Step k (1-indexed) of an N-step trajectory:
    R_k = gamma^(N-k) * outcome. Returns array of length num_steps (index 0 = step 1).

    NOTE: this is the AppWorld-offline simplification documented in the spec —
    the paper's real step return also carries per-step invalid-action penalties,
    which the frozen binary rollouts do not provide.
    """
    if num_steps <= 0:
        return np.zeros(0, dtype=np.float32)
    exps = np.arange(num_steps - 1, -1, -1, dtype=np.float32)  # [N-1, ..., 1, 0]
    return (gamma ** exps) * float(outcome)


def _build_step_groups(
    anchors_per_rollout: list[list[str]],
    *,
    enable_similarity: bool,
    threshold: float,
) -> list[list[int]]:
    """Anchor-state grouping (paper Eqs. 4, 6) across one task's rollouts.

    Input: anchors_per_rollout[i] = list of anchor strings for rollout i's steps.
    Output: same nested shape, each step replaced by an integer group id; steps
    sharing an anchor (exact, or similarity >= threshold) get the same id.

    Exact mode: hashmap on _to_hashable(anchor). Similarity mode: greedy
    clustering by SequenceMatcher ratio, matching the GiGPO reference.
    """
    if not enable_similarity:
        key_to_id: dict = {}
        out: list[list[int]] = []
        next_id = 0
        for anchors in anchors_per_rollout:
            ids = []
            for a in anchors:
                key = _to_hashable(a)
                if key not in key_to_id:
                    key_to_id[key] = next_id
                    next_id += 1
                ids.append(key_to_id[key])
            out.append(ids)
        return out

    # Similarity mode: greedy representative clustering.
    reps: list[str] = []
    out = []
    for anchors in anchors_per_rollout:
        ids = []
        for a in anchors:
            gid = None
            for j, rep in enumerate(reps):
                if _are_similar(a, rep, threshold):
                    gid = j
                    break
            if gid is None:
                gid = len(reps)
                reps.append(a)
            ids.append(gid)
        out.append(ids)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest grpo/tests/test_make_advantages.py -k "discounted_step_returns or build_step_groups" -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add grpo/preprocess/make_advantages.py grpo/tests/test_make_advantages.py
git commit -m "feat(preprocess): add GiGPO discounted returns + anchor-state grouping"
```

---

## Task 4: `compute_gigpo_advantage` (the joint A^E + ω·A^S)

**Files:**
- Modify: `grpo/preprocess/make_advantages.py`
- Test: `grpo/tests/test_make_advantages.py`

- [ ] **Step 1: Write the failing tests**

Add `compute_gigpo_advantage` to the import block in `grpo/tests/test_make_advantages.py`:

```python
from grpo.preprocess.make_advantages import (
    RolloutData,
    compute_serpo_advantage,
    compute_vanilla_advantage,
    compute_gigpo_advantage,
    _loo_norm,
    _to_hashable,
    _are_similar,
    _discounted_step_returns,
    _build_step_groups,
)
```

Add a local helper and tests:

```python
def _mock_gigpo_rollout(outcome, anchors):
    """2-step rollout: step1 tokens [2:5], step2 tokens [6:8]."""
    return RolloutData(
        task_id="t", seed=1,
        input_ids=list(range(10)), attention_mask=[1] * 10,
        response_mask=[0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        step_token_ranges=[(2, 5), (6, 8)],
        segments=[], outcome=float(outcome),
        step_anchor_obs=list(anchors),
    )


def test_gigpo_all_fail_group_is_zero():
    """All outcomes 0 -> A^E=0 and all step returns 0 -> A^S=0 -> all zero."""
    rollouts = [_mock_gigpo_rollout(0, ["A", "B"]) for _ in range(8)]
    compute_gigpo_advantage(rollouts, gamma=0.95, omega=1.0,
                            norm_mode="leave_one_out")
    for r in rollouts:
        assert np.allclose(r.token_adv, 0.0)


def test_gigpo_zero_outside_assistant():
    """response_mask=0 positions stay zero."""
    rollouts = [_mock_gigpo_rollout(1 if i < 2 else 0, ["A", "B"])
                for i in range(8)]
    compute_gigpo_advantage(rollouts, gamma=0.95, omega=1.0,
                            norm_mode="leave_one_out")
    for r in rollouts:
        for i, m in enumerate(r.response_mask):
            if m == 0:
                assert r.token_adv[i] == 0.0


def test_gigpo_additive_episode_plus_step():
    """token_adv on a step = A^E + omega*A^S; verify against hand computation.

    8 rollouts, 2 successes. Episode outcomes = [1,1,0,0,0,0,0,0], LOO mean=0.25
    -> A^E = outcome - 0.25. All rollouts share anchors ["A","B"] so step1 forms
    one group of 8 and step2 another group of 8.
    """
    rollouts = [_mock_gigpo_rollout(1 if i < 2 else 0, ["A", "B"])
                for i in range(8)]
    compute_gigpo_advantage(rollouts, gamma=0.95, omega=1.0,
                            norm_mode="leave_one_out")

    # Hand-compute expected A^E
    outcomes = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    ae = outcomes - outcomes.mean()  # leave_one_out

    # Step returns: step1 = gamma^1 * o, step2 = gamma^0 * o
    g = 0.95
    step1_returns = np.array([g * o for o in outcomes], dtype=np.float32)
    step2_returns = np.array([1.0 * o for o in outcomes], dtype=np.float32)
    as1 = step1_returns - step1_returns.mean()
    as2 = step2_returns - step2_returns.mean()

    for idx, r in enumerate(rollouts):
        # step1 token (index 2), step2 token (index 6)
        assert np.isclose(r.token_adv[2], ae[idx] + 1.0 * as1[idx], atol=1e-5)
        assert np.isclose(r.token_adv[6], ae[idx] + 1.0 * as2[idx], atol=1e-5)


def test_gigpo_singleton_step_group_has_zero_step_adv():
    """A step whose anchor is unique (group size 1) gets A^S=0, so its token_adv
    equals A^E alone."""
    # rollout 0 has a unique step-2 anchor "UNIQUE"; all others share "B"
    rollouts = [_mock_gigpo_rollout(1 if i < 2 else 0,
                                    ["A", "UNIQUE" if i == 0 else "B"])
                for i in range(8)]
    compute_gigpo_advantage(rollouts, gamma=0.95, omega=1.0,
                            norm_mode="leave_one_out")
    outcomes = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    ae0 = outcomes[0] - outcomes.mean()
    # rollout 0 step2 is alone in its group -> A^S = 0
    assert np.isclose(rollouts[0].token_adv[6], ae0, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest grpo/tests/test_make_advantages.py -k gigpo -v`
Expected: FAIL with `ImportError: cannot import name 'compute_gigpo_advantage'`

- [ ] **Step 3: Implement**

In `grpo/preprocess/make_advantages.py`, add after `compute_serpo_avg_advantage`:

```python
def compute_gigpo_advantage(
    rollouts: list[RolloutData],
    *,
    gamma: float = 0.95,
    omega: float = 1.0,
    norm_mode: str = "leave_one_out",
    enable_similarity: bool = False,
    similarity_thresh: float = 0.9,
) -> None:
    """GiGPO advantage (arXiv 2505.10978): A = A^E + omega * A^S.

    A^E: episode-level relative advantage = LOO-normalized binary outcome across
         the group, broadcast uniformly to a rollout's assistant tokens (Eq. 3).
    A^S: step-level relative advantage = LOO-normalized discounted step return
         within each anchor-state group, broadcast to that step's token span
         (Eqs. 4-8). Anchor = the env-output observation preceding the step.

    Operates on one task's group of rollouts; mutates r.token_adv in place.
    """
    # --- A^E: episode advantage (paper Eq. 3) ---
    outcomes = np.asarray([r.outcome for r in rollouts], dtype=np.float32)
    ae = _loo_norm(outcomes, mode=norm_mode)  # one scalar per rollout

    # --- anchor-state grouping (paper Eqs. 4, 6) ---
    anchors_per_rollout = [list(r.step_anchor_obs) for r in rollouts]
    group_ids = _build_step_groups(
        anchors_per_rollout,
        enable_similarity=enable_similarity,
        threshold=similarity_thresh,
    )

    # --- discounted step returns (paper Eq. 5, terminal-only) ---
    step_returns = [
        _discounted_step_returns(len(r.step_token_ranges), r.outcome, gamma)
        for r in rollouts
    ]

    # --- pool step returns by group id, LOO-normalize within group (Eq. 7) ---
    gid_to_returns: dict[int, list[float]] = {}
    for i, ids in enumerate(group_ids):
        for k, gid in enumerate(ids):
            gid_to_returns.setdefault(gid, []).append(float(step_returns[i][k]))
    gid_to_mean: dict[int, float] = {}
    gid_to_std: dict[int, float] = {}
    for gid, vals in gid_to_returns.items():
        arr = np.asarray(vals, dtype=np.float32)
        gid_to_mean[gid] = float(arr.mean())
        gid_to_std[gid] = float(arr.std())

    # --- write A = A^E + omega * A^S onto tokens ---
    for i, r in enumerate(rollouts):
        r.token_adv = np.zeros(len(r.input_ids), dtype=np.float32)
        # A^E uniform on assistant tokens
        mask = np.asarray(r.response_mask, dtype=bool)
        r.token_adv[mask] = ae[i]
        # A^S per step, added on top
        ids = group_ids[i]
        for k, (tok_start, tok_end) in enumerate(r.step_token_ranges):
            gid = ids[k]
            n_in_group = len(gid_to_returns[gid])
            if n_in_group <= 1:
                a_s = 0.0  # singleton group -> no relative signal
            else:
                centered = float(step_returns[i][k]) - gid_to_mean[gid]
                if norm_mode == "std":
                    a_s = centered / (gid_to_std[gid] + EPS)
                else:  # leave_one_out
                    a_s = centered
            r.token_adv[tok_start:tok_end] += omega * a_s
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest grpo/tests/test_make_advantages.py -k gigpo -v`
Expected: all PASS

- [ ] **Step 5: Run the full advantages suite**

Run: `python -m pytest grpo/tests/test_make_advantages.py -v`
Expected: all PASS (existing serpo/vanilla tests unaffected)

- [ ] **Step 6: Commit**

```bash
git add grpo/preprocess/make_advantages.py grpo/tests/test_make_advantages.py
git commit -m "feat(preprocess): add compute_gigpo_advantage (A^E + omega*A^S)"
```

---

## Task 5: Emit `step_anchor_obs` from `tokenize_trajectory`

**Files:**
- Modify: `grpo/preprocess/tokenize_trajectory.py` (the message-walk loop, lines 185-220)
- Test: `grpo/tests/test_tokenize_trajectory.py`

**Context:** `tokenize_trajectory` walks `augmented` messages; assistant messages after `prefix_end` become steps. The anchor for step k is the *content of the user message immediately preceding that assistant message* (the env output / task prefix for step 1). We capture it during the same walk.

- [ ] **Step 1: Write the failing test**

Read `grpo/tests/test_tokenize_trajectory.py` first to match its tokenizer fixture/style, then add a test that asserts the returned dict contains `step_anchor_obs` with length == num_steps and that each entry is the preceding user message's content. Add:

```python
def test_step_anchor_obs_length_and_content(qwen_tokenizer, tmp_path):
    """step_anchor_obs has one entry per step = the preceding user message."""
    # Build a minimal lm_calls.jsonl with: user(prefix), assistant(step1),
    # user(env-out-1), assistant(step2). Reuse the same lm_calls fixture builder
    # used by the existing tests in this file.
    lm_calls = _write_lm_calls(  # existing helper in this test module
        tmp_path,
        messages=[
            {"role": "user", "content":
                "Using these APIs, now generate code to solve the actual task: do X"},
            {"role": "assistant", "content": "step one code"},
            {"role": "user", "content": "Output:\n```\nenv result 1\n```"},
        ],
        output_message={"role": "assistant", "content": "step two code"},
    )
    result = tokenize_trajectory(lm_calls, qwen_tokenizer)
    assert len(result["step_anchor_obs"]) == result["num_steps"]
    assert len(result["step_anchor_obs"]) == 2
    # step 2's anchor is the env output preceding it
    assert "env result 1" in result["step_anchor_obs"][1]
    # step 1's anchor is the task-instruction prefix
    assert "solve the actual task" in result["step_anchor_obs"][0]
```

> If `test_tokenize_trajectory.py` has no `_write_lm_calls`/`qwen_tokenizer` helper, adapt the test to whatever fixture the existing tests use (e.g. a real `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")` and an inline jsonl write). The assertions on `step_anchor_obs` are the part that matters.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grpo/tests/test_tokenize_trajectory.py::test_step_anchor_obs_length_and_content -v`
Expected: FAIL with `KeyError: 'step_anchor_obs'`

- [ ] **Step 3: Capture the anchor during the message walk**

In `grpo/preprocess/tokenize_trajectory.py`, inside `tokenize_trajectory`, modify the walk loop (currently building `response_mask` and `step_ranges`). Add a `step_anchor_obs` list and record the previous non-empty message content when an assistant step is found.

Replace the loop body (the `for i, msg in enumerate(augmented):` block, lines ~188-208) so it also tracks the last-seen message content and appends an anchor per step:

```python
    response_mask = [0] * len(input_ids)
    step_ranges: list[tuple[int, int]] = []
    step_anchor_obs: list[str] = []
    cursor = 0
    prev_content = ""  # content of the message immediately before current
    for i, msg in enumerate(augmented):
        content = msg.get("content", "")
        target = content.rstrip()
        if not target:
            continue
        pos = chat_str.find(target, cursor)
        if pos < 0:
            pos = chat_str.find(content, cursor)
            target = content
        if pos < 0:
            raise RuntimeError(
                f"message {i} ({msg.get('role')}) content not found in render"
            )
        cstart, cend = pos, pos + len(target)
        cursor = cend
        if i > prefix_end and msg.get("role") == "assistant":
            tstart = _char_to_token(cstart, start=True)
            tend = _char_to_token(cend, start=True)
            for k in range(tstart, tend):
                response_mask[k] = 1
            step_ranges.append((tstart, tend))
            step_anchor_obs.append(prev_content)
        prev_content = content
```

Then add `step_anchor_obs` to the returned dict (the final `return {...}`):

```python
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "response_mask": response_mask,
        "step_token_ranges": step_ranges,
        "step_anchor_obs": step_anchor_obs,
        "num_steps": len(step_ranges),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest grpo/tests/test_tokenize_trajectory.py::test_step_anchor_obs_length_and_content -v`
Expected: PASS

- [ ] **Step 5: Run the full tokenizer suite**

Run: `python -m pytest grpo/tests/test_tokenize_trajectory.py -v`
Expected: all PASS (the new field is additive; existing assertions on input_ids/response_mask/step_token_ranges unchanged)

- [ ] **Step 6: Commit**

```bash
git add grpo/preprocess/tokenize_trajectory.py grpo/tests/test_tokenize_trajectory.py
git commit -m "feat(preprocess): emit step_anchor_obs from tokenize_trajectory"
```

---

## Task 6: Wire `gigpo` method into `build_dataset.py`

**Files:**
- Modify: `grpo/preprocess/build_dataset.py` (RolloutData construction ~144-155; method dispatch ~157-164; CLI ~349-379; fail-only guard)
- Test: `grpo/tests/test_build_dataset.py`

- [ ] **Step 1: Write the failing test**

Read `grpo/tests/test_build_dataset.py` to match its fixtures, then add:

```python
def test_gigpo_failonly_is_rejected():
    """GiGPO + failonly is degenerate (binary all-zero) -> must error."""
    import pytest
    from grpo.preprocess.build_dataset import run_build
    with pytest.raises(ValueError, match="gigpo.*failonly|failonly.*gigpo"):
        run_build(
            method="gigpo", condition="failonly",
            rollout_dir=Path("/nonexistent"), joint_dir=Path("/nonexistent"),
            output_dir=Path("/tmp/x"), outcome_type="binary",
        )
```

If `test_build_dataset.py` already exercises `build_dataset_for_task_group` with mock inputs, also add a test that passing `method="gigpo"` with a 2-rollout group sets nonzero `advantages` (mirror the existing serpo/vanilla group test, adding `step_anchor_obs` to each tokenize result mock).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grpo/tests/test_build_dataset.py::test_gigpo_failonly_is_rejected -v`
Expected: FAIL (no guard yet → wrong/no exception)

- [ ] **Step 3: Add the fail-only guard + method choices**

In `grpo/preprocess/build_dataset.py`, import the new function (top, with the other advantage imports):

```python
from grpo.preprocess.make_advantages import (
    RolloutData,
    compute_serpo_advantage,
    compute_serpo_avg_advantage,
    compute_vanilla_advantage,
    compute_gigpo_advantage,
)
```

In `run_build`, immediately after the existing `if condition not in ("failonly", "full")` validation, add:

```python
    if method == "gigpo" and condition == "failonly":
        raise ValueError(
            "method=gigpo with condition=failonly is degenerate: binary reward "
            "is 0 for every fail-only rollout, so A^E and A^S are all zero. "
            "Use condition=full for GiGPO."
        )
```

- [ ] **Step 4: Pass anchor obs into RolloutData and dispatch gigpo**

In `build_dataset_for_task_group`, add `step_anchor_obs` to the `RolloutData(...)` construction (after `segments=...`):

```python
                step_anchor_obs=tok_result.get("step_anchor_obs", []),
```

Add the dispatch branch (after the `serpo_avg` branch, before the `else: raise`):

```python
    elif method == "gigpo":
        compute_gigpo_advantage(rollouts)
```

> GiGPO uses its default hyperparameters (γ=0.95, ω=1, leave_one_out) here. CLI overrides are threaded in Step 5.

- [ ] **Step 5: Add CLI args and thread them through**

Add `"gigpo"` to the `--method` choices in `main()`:

```python
    ap.add_argument("--method", choices=["vanilla", "serpo", "serpo_avg", "gigpo"], required=True)
```

Add GiGPO CLI args after `--tokenizer-name`:

```python
    ap.add_argument("--gamma", type=float, default=0.95,
                    help="GiGPO discount factor (paper: 0.95).")
    ap.add_argument("--omega", type=float, default=1.0,
                    help="GiGPO step-advantage weight (paper: 1.0).")
    ap.add_argument("--gigpo-norm-mode", choices=["leave_one_out", "std"],
                    default="leave_one_out",
                    help="GiGPO normalization (paper F_norm=1 default).")
    ap.add_argument("--enable-similarity", action="store_true",
                    help="GiGPO: cluster anchors by similarity instead of exact match.")
    ap.add_argument("--similarity-thresh", type=float, default=0.9,
                    help="GiGPO similarity threshold (paper: 0.9).")
```

Thread them: change `build_dataset_for_task_group` and `run_build` to accept a `gigpo_kwargs: dict | None = None` parameter, and in the dispatch branch use it:

```python
    elif method == "gigpo":
        compute_gigpo_advantage(rollouts, **(gigpo_kwargs or {}))
```

In `run_build`, accept `gigpo_kwargs` and pass it to `build_dataset_for_task_group(...)`. In `main()`, build it:

```python
    gigpo_kwargs = {
        "gamma": args.gamma, "omega": args.omega,
        "norm_mode": args.gigpo_norm_mode,
        "enable_similarity": args.enable_similarity,
        "similarity_thresh": args.similarity_thresh,
    } if args.method == "gigpo" else None
    run_build(
        args.method, args.condition, args.rollout_dir, args.joint_dir,
        args.output_dir, outcome_type=args.outcome_type,
        tokenizer_name=args.tokenizer_name, gigpo_kwargs=gigpo_kwargs,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest grpo/tests/test_build_dataset.py -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add grpo/preprocess/build_dataset.py grpo/tests/test_build_dataset.py
git commit -m "feat(preprocess): wire gigpo method + CLI into build_dataset"
```

---

## Task 7: Smoke-test on real AppWorld rollouts (no GPU)

**Files:** none (verification only). Produces `grpo/data/round0_9b/gigpo_full_binary.parquet`.

- [ ] **Step 1: Build the GiGPO parquet from existing rollouts**

Run:

```bash
python -m grpo.preprocess.build_dataset \
  --method gigpo --condition full --outcome-type binary \
  --rollout-dir appworld/experiments/outputs/rollout/round0_9b \
  --joint-dir rubric_reward/results/rollout/round0_9b/KS_baseline \
  --output-dir grpo/data/round0_9b \
  --tokenizer-name Qwen/Qwen3.5-9B
```

Expected: log lines ending with `done: N groups written ... parquet: grpo/data/round0_9b/gigpo_full_binary.parquet`. (joint-dir is unused by gigpo but the arg is required; any existing path is fine.)

- [ ] **Step 2: Sanity-check the parquet**

Run:

```bash
python -c "
import pyarrow.parquet as pq, numpy as np
t = pq.read_table('grpo/data/round0_9b/gigpo_full_binary.parquet').to_pandas()
print('rows:', len(t))
advs = np.concatenate([np.asarray(a) for a in t['advantages']])
print('adv nonzero frac:', float((advs != 0).mean()))
print('adv min/mean/max:', advs.min(), advs.mean(), advs.max())
assert np.isfinite(advs).all(), 'non-finite advantage!'
assert (advs != 0).any(), 'all-zero advantages — grouping/outcome wiring broke'
print('OK')
"
```

Expected: nonzero fraction > 0, finite advantages, prints `OK`. (On the `full` set there are successes, so A^E varies and A^S is nonzero where anchors group.)

- [ ] **Step 3: Commit the verification note**

No code to commit. If a data/README log exists for the round, append a one-line note that `gigpo_full_binary.parquet` was built; otherwise skip.

---

## Task 8: Training config + launch script

**Files:**
- Create: `grpo/trainer/config/gigpo_9b_full_binary_v3i.yaml`
- Create: `grpo/scripts/train_gigpo_9b_full_binary_v3i.sh`

**Context:** Clone the vanilla full-binary v3i config, changing only the parquet path, output dir, and wandb run name. Read `grpo/trainer/config/vanilla_9b_full_binary_v3i.yaml` first to copy its exact hyperparameters (lr, K_steps, guards, LoRA target modules) — those are shared across all arms for a fair comparison.

- [ ] **Step 1: Create the config**

Read `grpo/trainer/config/vanilla_9b_full_binary_v3i.yaml`, copy it verbatim, then change exactly three fields:

```yaml
parquet_path: grpo/data/round0_9b/gigpo_full_binary.parquet
output_dir: grpo/ckpts/gigpo_9b_full_binary_v3i/round1
wandb_run_name: gigpo_9b_full_binary_v3i
```

All other fields (model_name, K_steps, mini_batch_size, micro_batch_size, precache_micro_batch_size, lr, guards, lora_*, outlier/mask settings) must match the vanilla config exactly. Save to `grpo/trainer/config/gigpo_9b_full_binary_v3i.yaml`.

- [ ] **Step 2: Create the launch script**

Read `grpo/scripts/train_vanilla_9b_full_binary_v3i.sh`, copy it, change the `--config` path to `grpo/trainer/config/gigpo_9b_full_binary_v3i.yaml` (and any run-name echo). Save to `grpo/scripts/train_gigpo_9b_full_binary_v3i.sh` and `chmod +x` it.

- [ ] **Step 3: Validate the config loads (no GPU)**

Run:

```bash
python -c "
import yaml
from grpo.trainer.offline_trainer import GRPOConfig
d = yaml.safe_load(open('grpo/trainer/config/gigpo_9b_full_binary_v3i.yaml'))
if isinstance(d.get('lora_target_modules'), list):
    d['lora_target_modules'] = tuple(d['lora_target_modules'])
cfg = GRPOConfig(**d)
print('config OK:', cfg.parquet_path, '->', cfg.output_dir)
"
```

Expected: prints `config OK: grpo/data/round0_9b/gigpo_full_binary.parquet -> grpo/ckpts/gigpo_9b_full_binary_v3i/round1`

- [ ] **Step 4: Commit**

```bash
git add grpo/trainer/config/gigpo_9b_full_binary_v3i.yaml grpo/scripts/train_gigpo_9b_full_binary_v3i.sh
git commit -m "feat(trainer): GiGPO full-binary v3i config + launch script"
```

- [ ] **Step 5: Stop — training is a GPU run, requires explicit permission**

Do NOT launch training. The launch script is ready; per the GPU-permission rule, ask the user before running `train_gigpo_9b_full_binary_v3i.sh` (it occupies GPU memory on the shared server).

---

## Self-Review

**Spec coverage:**
- Anchor obs extraction → Task 5 ✓
- `compute_gigpo_advantage` (A^E + ω·A^S, γ=0.95, ω=1, LOO) → Tasks 2-4 ✓
- Anchor-state grouping (exact + similarity 0.9) → Task 3 ✓
- Discounted terminal step return → Task 3 ✓
- build_dataset gigpo method + CLI + fail-only guard → Task 6 ✓
- `RolloutData.step_anchor_obs` field → Task 1 ✓
- Config + script (full/binary, clone of vanilla) → Task 8 ✓
- Edge cases: all-fail (Task 4), singleton cluster (Task 4), zero-outside-assistant (Task 4) ✓
- Tests in `test_make_advantages.py` / `test_tokenize_trajectory.py` / `test_build_dataset.py` ✓
- GPU run deferred to user permission → Task 8 Step 5 ✓
- Out of scope (trainer, continuous, additive-serpo) → not in plan ✓

**Placeholder scan:** Task 5 and Task 6 tests say "read the existing test file to match fixtures" — this is necessary because those test modules' fixture helpers weren't read during planning; the *assertions* that matter are fully specified. Task 8 says "copy the vanilla config/script verbatim" with the exact three fields to change — concrete. No TBD/TODO left.

**Type consistency:** `compute_gigpo_advantage` signature (kwargs gamma/omega/norm_mode/enable_similarity/similarity_thresh) is identical across Task 4 (def), Task 6 (call via gigpo_kwargs). `_build_step_groups` returns `list[list[int]]` consistently. `step_anchor_obs` key name identical across tokenizer (Task 5), RolloutData (Task 1), build_dataset (Task 6). `_loo_norm(mode=...)` vs the inline std/loo branch in `compute_gigpo_advantage` use the same mode strings (`leave_one_out`/`std`). ✓
