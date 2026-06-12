# AppWorld Trajectory Comparison Video — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained HTML page that plays two AppWorld agent trajectories side-by-side per task (Vanilla-GRPO fails → SeRPO succeeds), with chat bubbles (agent code + env response) on the left and a mock app UI on the right, for two single-app tasks selectable from a start screen.

**Architecture:** A Python build script parses the four `environment_io.md` files (2 tasks × 2 models) into per-(task,model) JSON, classifying each step's `apis.<app>.<api>` call. A pure-Python per-app deriver bakes a `ui_state` into each step (mapped for Venmo + Phone, generic fallback otherwise). A static `player.html` + `player.js` loads the generated JSON, shows a task-selector start screen, and on click animates both models on one shared clock.

**Tech Stack:** Python 3.11 (stdlib only — `re`, `json`, `pathlib`, `argparse`), `pytest` for tests, plain HTML/CSS/vanilla JS (no framework, no build step).

**Spec:** [docs/superpowers/specs/2026-06-12-trajectory-comparison-video-design.md](../specs/2026-06-12-trajectory-comparison-video-design.md)

---

## File Structure

```
viz/
  __init__.py                  # makes viz a package so tests can import
  parse_io.py                  # parse environment_io.md → list[Step]  (Task 1)
  app_state.py                 # derive_ui_state(app, api, output, prev) (Task 2-3)
  build_trajectory_json.py     # orchestrator: read files → write data/*.json (Task 4)
  player.html                  # start screen + stage markup (Task 5)
  player.css                   # styling (Task 6)
  player.js                    # selector + timeline animation (Task 6)
  data/                        # generated JSON (committed; small)
    manifest.json
    552869a_2.vanilla.json
    552869a_2.serpo.json
    31dc501_2.vanilla.json
    31dc501_2.serpo.json
  tests/
    __init__.py
    fixtures/sample_io.md       # tiny hand-written environment_io.md
    test_parse_io.py            # Task 1
    test_app_state.py           # Task 2-3
    test_build.py               # Task 4
```

**Data inputs (read-only, already on disk):**
- Trajectory: `appworld/experiments/outputs/eval/dev/<config>/tasks/<task_id>/logs/environment_io.md`
  - vanilla config = `vanilla60_test_test_normal`, serpo config = `serpoavg60_test_test_normal`
- Pass/fail: `.../tasks/<task_id>/evaluation/report.md` (passed iff `Num Failed Tests : 0`)
- Instruction: `appworld/data/tasks/<task_id>/specs.json` → key `instruction`

**Featured tasks:** `552869a_2` (venmo), `31dc501_2` (phone).

---

## Task 1: Parse environment_io.md into steps

**Files:**
- Create: `viz/__init__.py` (empty)
- Create: `viz/tests/__init__.py` (empty)
- Create: `viz/tests/fixtures/sample_io.md`
- Create: `viz/parse_io.py`
- Test: `viz/tests/test_parse_io.py`

- [ ] **Step 1: Create package + test init files**

Create `viz/__init__.py` and `viz/tests/__init__.py` as empty files.

- [ ] **Step 2: Create the fixture**

Create `viz/tests/fixtures/sample_io.md` with exactly this content (mirrors the real format — leading blank line, `### Environment Interaction N`, dashed rule, ```` ```python ```` fence, blank line, bare ```` ``` ```` output fence):

````markdown

### Environment Interaction 1
----------------------------------------------------------------------------
```python
print(apis.api_docs.show_api_descriptions(app_name='venmo'))
```

```
[
 {"name": "login"}
]
```


### Environment Interaction 2
----------------------------------------------------------------------------
```python
login_result = apis.venmo.login(username='a@b.com', password='x')
print(login_result)
```

```
{"access_token": "TOK"}
```


### Environment Interaction 3
----------------------------------------------------------------------------
```python
apis.supervisor.complete_task(answer=144.0)
```

```
Execution successful.
```
````

- [ ] **Step 3: Write the failing test**

Create `viz/tests/test_parse_io.py`:

```python
from pathlib import Path
from viz.parse_io import parse_io, Step

FIXTURE = Path(__file__).parent / "fixtures" / "sample_io.md"


def test_parses_three_steps():
    steps = parse_io(FIXTURE.read_text())
    assert len(steps) == 3
    assert all(isinstance(s, Step) for s in steps)


def test_step_numbers_are_sequential():
    steps = parse_io(FIXTURE.read_text())
    assert [s.step for s in steps] == [1, 2, 3]


def test_code_and_output_split():
    steps = parse_io(FIXTURE.read_text())
    assert steps[1].code == "login_result = apis.venmo.login(username='a@b.com', password='x')\nprint(login_result)"
    assert steps[1].output == '{"access_token": "TOK"}'


def test_classifies_app_and_api():
    steps = parse_io(FIXTURE.read_text())
    # Step 1 calls api_docs.show_api_descriptions
    assert steps[0].app == "api_docs"
    assert steps[0].api == "show_api_descriptions"
    # Step 2 calls venmo.login
    assert steps[1].app == "venmo"
    assert steps[1].api == "login"
    # Step 3 calls supervisor.complete_task
    assert steps[2].app == "supervisor"
    assert steps[2].api == "complete_task"


def test_empty_code_step_has_none_app():
    # A step whose code calls no api.* should have app/api = None
    md = "\n### Environment Interaction 1\n---\n```python\nx = 1 + 1\n```\n\n```\n2\n```\n"
    steps = parse_io(md)
    assert steps[0].app is None
    assert steps[0].api is None
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /data/minjeong/Autonomous_agent && python -m pytest viz/tests/test_parse_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'viz.parse_io'`

- [ ] **Step 5: Write minimal implementation**

Create `viz/parse_io.py`:

```python
"""Parse AppWorld environment_io.md transcripts into structured steps."""
import re
from dataclasses import dataclass


@dataclass
class Step:
    step: int
    code: str
    output: str
    app: str | None      # e.g. "venmo", or None if the code calls no apis.* method
    api: str | None       # e.g. "login"


# A block starts at "### Environment Interaction N" and runs until the next one (or EOF).
_BLOCK_RE = re.compile(
    r"### Environment Interaction (\d+)\s*\n"   # header + number
    r".*?"                                        # dashed rule line(s)
    r"```python\s*\n(.*?)\n```"                  # code fence -> group 2
    r".*?"                                        # gap (blank line)
    r"```\s*\n(.*?)\n```",                        # output fence -> group 3
    re.DOTALL,
)

# Last apis.<app>.<api>( call in the code is treated as the primary/state-changing one.
_API_RE = re.compile(r"apis\.([a-z_]+)\.([a-z_]+)\s*\(")


def _classify(code: str) -> tuple[str | None, str | None]:
    matches = _API_RE.findall(code)
    if not matches:
        return None, None
    app, api = matches[-1]
    return app, api


def parse_io(text: str) -> list[Step]:
    steps: list[Step] = []
    for m in _BLOCK_RE.finditer(text):
        num = int(m.group(1))
        code = m.group(2).strip("\n")
        output = m.group(3).strip("\n")
        app, api = _classify(code)
        steps.append(Step(step=num, code=code, output=output, app=app, api=api))
    return steps
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /data/minjeong/Autonomous_agent && python -m pytest viz/tests/test_parse_io.py -v`
Expected: PASS (5 passed)

- [ ] **Step 7: Verify against real data**

Run: `cd /data/minjeong/Autonomous_agent && python -c "from viz.parse_io import parse_io; from pathlib import Path; s=parse_io(Path('appworld/experiments/outputs/eval/dev/serpoavg60_test_test_normal/tasks/552869a_2/logs/environment_io.md').read_text()); print(len(s), 'steps'); print(s[-1].app, s[-1].api)"`
Expected: `8 steps` and `supervisor complete_task`

- [ ] **Step 8: Commit**

```bash
cd /data/minjeong/Autonomous_agent
git add viz/__init__.py viz/tests/__init__.py viz/tests/fixtures/sample_io.md viz/parse_io.py viz/tests/test_parse_io.py
git commit -m "feat(viz): parse environment_io.md into classified steps"
```

---

## Task 2: App-state deriver — fallback + Venmo

**Files:**
- Create: `viz/app_state.py`
- Test: `viz/tests/test_app_state.py`

The deriver returns a small dict describing what the right-hand mock UI should render after a
step. Shape: `{"kind": <str>, "title": <str>, "rows": [...], "answer": <any>}` (fields present
per kind). `kind` ∈ `{"docs", "login", "venmo_transactions", "phone_alarms", "result", "api_call"}`.

- [ ] **Step 1: Write the failing test (fallback + venmo)**

Create `viz/tests/test_app_state.py`:

```python
import json
from viz.app_state import derive_ui_state


def test_fallback_for_unmapped_api():
    st = derive_ui_state(app="venmo", api="like_transaction", output="{}", prev=None)
    assert st["kind"] == "api_call"
    assert "like_transaction" in st["title"]


def test_api_docs_is_docs_state():
    st = derive_ui_state(app="api_docs", api="show_api_descriptions", output="[]", prev=None)
    assert st["kind"] == "docs"


def test_none_app_keeps_prev_state():
    prev = {"kind": "login", "title": "Venmo"}
    st = derive_ui_state(app=None, api=None, output="2023-01-01", prev=prev)
    assert st == prev  # a pure-compute step doesn't change the app screen


def test_venmo_login():
    out = '{"access_token": "TOK", "token_type": "Bearer"}'
    st = derive_ui_state(app="venmo", api="login", output=out, prev=None)
    assert st["kind"] == "login"
    assert st["title"].lower().startswith("venmo")


def test_venmo_show_transactions_renders_rows():
    out = json.dumps([
        {"amount": 50.0, "description": "Electricity bill",
         "created_at": "2023-03-01T00:00:00",
         "sender": {"name": "Jose"}, "receiver": {"name": "PowerCo"}},
        {"amount": 12.0, "description": "Coffee",
         "created_at": "2023-03-02T00:00:00",
         "sender": {"name": "Jose"}, "receiver": {"name": "Cafe"}},
    ])
    st = derive_ui_state(app="venmo", api="show_transactions", output=out, prev=None)
    assert st["kind"] == "venmo_transactions"
    assert len(st["rows"]) == 2
    assert st["rows"][0]["amount"] == 50.0
    assert st["rows"][0]["description"] == "Electricity bill"


def test_complete_task_is_result():
    st = derive_ui_state(app="supervisor", api="complete_task",
                         output="Execution successful.", prev=None)
    assert st["kind"] == "result"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/minjeong/Autonomous_agent && python -m pytest viz/tests/test_app_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'viz.app_state'`

- [ ] **Step 3: Write minimal implementation**

Create `viz/app_state.py`:

```python
"""Derive a mock-app UI state from a single trajectory step.

Pure functions: derive_ui_state(app, api, output, prev) -> dict.
Mapped for venmo + phone; everything else falls back to a generic 'api_call' state.
A step that calls no API (app is None) leaves the screen unchanged (returns prev).
"""
import json
from typing import Any


def _try_json(output: str) -> Any:
    try:
        return json.loads(output)
    except (ValueError, TypeError):
        return None


def derive_ui_state(app: str | None, api: str | None, output: str,
                    prev: dict | None) -> dict:
    # Pure-compute / no-API step: screen does not change.
    if app is None:
        return prev if prev is not None else {"kind": "idle", "title": ""}

    if app == "api_docs":
        return {"kind": "docs", "title": "📖 Reading API docs"}

    if api == "complete_task":
        return {"kind": "result", "title": "✅ Task submitted"}

    if app == "venmo":
        return _venmo(api, output, prev)
    if app == "phone":
        return _phone(api, output, prev)

    # Generic fallback for any unmapped (app, api).
    return {"kind": "api_call", "title": f"⚙️ {app}.{api}()"}


def _venmo(api: str | None, output: str, prev: dict | None) -> dict:
    if api == "login":
        return {"kind": "login", "title": "Venmo — signed in"}
    if api == "show_transactions":
        data = _try_json(output)
        rows = []
        if isinstance(data, list):
            for tx in data:
                rows.append({
                    "amount": tx.get("amount"),
                    "description": tx.get("description", ""),
                    "created_at": tx.get("created_at", ""),
                    "sender": (tx.get("sender") or {}).get("name", ""),
                    "receiver": (tx.get("receiver") or {}).get("name", ""),
                })
        return {"kind": "venmo_transactions", "title": "Venmo — transactions",
                "rows": rows}
    return {"kind": "api_call", "title": f"⚙️ venmo.{api}()"}


def _phone(api: str | None, output: str, prev: dict | None) -> dict:
    # Implemented in Task 3.
    return {"kind": "api_call", "title": f"⚙️ phone.{api}()"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/minjeong/Autonomous_agent && python -m pytest viz/tests/test_app_state.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
cd /data/minjeong/Autonomous_agent
git add viz/app_state.py viz/tests/test_app_state.py
git commit -m "feat(viz): app-state deriver with fallback + venmo mapping"
```

---

## Task 3: App-state deriver — Phone (alarms)

**Files:**
- Modify: `viz/app_state.py` (replace the `_phone` stub)
- Test: `viz/tests/test_app_state.py` (add cases)

The phone task (`31dc501_2`) calls `show_alarms` then `update_alarm`. The mock UI shows an
alarms list and animates the targeted alarm's snooze value changing.

- [ ] **Step 1: Add failing tests for phone**

Append to `viz/tests/test_app_state.py`:

```python
def test_phone_show_alarms_renders_rows():
    out = json.dumps([
        {"alarm_id": 1, "label": "Weekend wake up", "time": "08:00",
         "snooze_minutes": 5, "enabled": True},
        {"alarm_id": 2, "label": "Workday", "time": "06:30",
         "snooze_minutes": 10, "enabled": True},
    ])
    st = derive_ui_state(app="phone", api="show_alarms", output=out, prev=None)
    assert st["kind"] == "phone_alarms"
    assert len(st["rows"]) == 2
    assert st["rows"][0]["label"] == "Weekend wake up"
    assert st["rows"][0]["snooze_minutes"] == 5


def test_phone_update_alarm_marks_changed():
    prev = {"kind": "phone_alarms", "title": "Phone — alarms", "rows": [
        {"alarm_id": 1, "label": "Weekend wake up", "time": "08:00",
         "snooze_minutes": 5, "enabled": True},
    ]}
    # update_alarm output echoes the updated alarm.
    out = json.dumps({"alarm_id": 1, "label": "Weekend wake up",
                      "time": "08:00", "snooze_minutes": 15, "enabled": True})
    st = derive_ui_state(app="phone", api="update_alarm", output=out, prev=prev)
    assert st["kind"] == "phone_alarms"
    row = next(r for r in st["rows"] if r["alarm_id"] == 1)
    assert row["snooze_minutes"] == 15
    assert row["changed"] is True
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `cd /data/minjeong/Autonomous_agent && python -m pytest viz/tests/test_app_state.py -k phone -v`
Expected: FAIL — `update_alarm` returns `api_call` kind, not `phone_alarms`.

- [ ] **Step 3: Replace the `_phone` stub with real implementation**

In `viz/app_state.py`, replace the entire `_phone` function with:

```python
def _phone(api: str | None, output: str, prev: dict | None) -> dict:
    if api == "login":
        return {"kind": "login", "title": "Phone — signed in"}
    if api == "show_alarms":
        data = _try_json(output)
        rows = []
        if isinstance(data, list):
            for a in data:
                rows.append({
                    "alarm_id": a.get("alarm_id"),
                    "label": a.get("label", ""),
                    "time": a.get("time", ""),
                    "snooze_minutes": a.get("snooze_minutes"),
                    "enabled": a.get("enabled", True),
                    "changed": False,
                })
        return {"kind": "phone_alarms", "title": "Phone — alarms", "rows": rows}
    if api == "update_alarm":
        updated = _try_json(output)
        # Start from the previous alarms list if present, else a fresh single-row list.
        prev_rows = (prev or {}).get("rows", []) if isinstance(prev, dict) else []
        rows = [dict(r) for r in prev_rows]  # shallow copy
        if isinstance(updated, dict):
            uid = updated.get("alarm_id")
            found = False
            for r in rows:
                r["changed"] = False
                if r.get("alarm_id") == uid:
                    r["snooze_minutes"] = updated.get("snooze_minutes",
                                                       r.get("snooze_minutes"))
                    r["changed"] = True
                    found = True
            if not found and uid is not None:
                rows.append({
                    "alarm_id": uid, "label": updated.get("label", ""),
                    "time": updated.get("time", ""),
                    "snooze_minutes": updated.get("snooze_minutes"),
                    "enabled": updated.get("enabled", True), "changed": True,
                })
        return {"kind": "phone_alarms", "title": "Phone — alarms", "rows": rows}
    return {"kind": "api_call", "title": f"⚙️ phone.{api}()"}
```

- [ ] **Step 4: Run full app_state tests**

Run: `cd /data/minjeong/Autonomous_agent && python -m pytest viz/tests/test_app_state.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
cd /data/minjeong/Autonomous_agent
git add viz/app_state.py viz/tests/test_app_state.py
git commit -m "feat(viz): phone alarms mapping (show_alarms + update_alarm)"
```

---

## Task 4: Build script — read files → write JSON

**Files:**
- Create: `viz/build_trajectory_json.py`
- Test: `viz/tests/test_build.py`

Orchestrates: for each (task, model) read `environment_io.md`, parse, derive `ui_state` per
step (threading `prev`), read instruction from `specs.json`, read pass/fail from `report.md`,
and write `viz/data/<task>.<model>.json` + a `viz/data/manifest.json`.

- [ ] **Step 1: Write the failing test**

Create `viz/tests/test_build.py`:

```python
from viz.build_trajectory_json import passed_from_report, build_step_records


def test_passed_from_report_true():
    md = "Num Passed Tests : 5\nNum Failed Tests : 0\nNum Total  Tests : 5\n"
    assert passed_from_report(md) is True


def test_passed_from_report_false():
    md = "Num Passed Tests : 4\nNum Failed Tests : 1\nNum Total  Tests : 5\n"
    assert passed_from_report(md) is False


def test_build_step_records_threads_prev_state():
    # Two steps: show_alarms then a pure-compute step. The compute step must
    # inherit the alarms screen (prev), not reset to idle.
    io = (
        "\n### Environment Interaction 1\n---\n```python\n"
        "a = apis.phone.show_alarms(access_token='t')\n```\n\n```\n"
        '[{"alarm_id": 1, "label": "X", "time": "08:00", "snooze_minutes": 5}]\n```\n'
        "\n### Environment Interaction 2\n---\n```python\nx = 1 + 1\n```\n\n```\n2\n```\n"
    )
    records = build_step_records(io)
    assert records[0]["ui_state"]["kind"] == "phone_alarms"
    assert records[1]["ui_state"]["kind"] == "phone_alarms"  # inherited prev
    assert records[1]["code"] == "x = 1 + 1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/minjeong/Autonomous_agent && python -m pytest viz/tests/test_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'viz.build_trajectory_json'`

- [ ] **Step 3: Write the implementation**

Create `viz/build_trajectory_json.py`:

```python
"""Build trajectory JSON for the comparison-video player.

Reads the four environment_io.md files (2 tasks x 2 models), the task
instructions, and pass/fail reports; writes per-(task,model) JSON plus a
manifest into viz/data/.
"""
import argparse
import json
import re
from pathlib import Path

from viz.parse_io import parse_io
from viz.app_state import derive_ui_state

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "appworld/experiments/outputs/eval/dev"
TASKS = ROOT / "appworld/data/tasks"

CONFIGS = {"vanilla": "vanilla60_test_test_normal",
           "serpo": "serpoavg60_test_test_normal"}

FEATURED = [
    {"task_id": "552869a_2", "app": "venmo"},
    {"task_id": "31dc501_2", "app": "phone"},
]

_FAILED_RE = re.compile(r"Num Failed Tests\s*:\s*(\d+)")


def passed_from_report(report_md: str) -> bool | None:
    m = _FAILED_RE.search(report_md)
    if not m:
        return None
    return int(m.group(1)) == 0


def build_step_records(io_text: str) -> list[dict]:
    steps = parse_io(io_text)
    records: list[dict] = []
    prev = None
    for s in steps:
        ui = derive_ui_state(s.app, s.api, s.output, prev)
        prev = ui
        records.append({
            "step": s.step, "code": s.code, "output": s.output,
            "app": s.app, "api": s.api, "ui_state": ui,
        })
    return records


def _instruction(task_id: str) -> str:
    spec = json.loads((TASKS / task_id / "specs.json").read_text())
    return spec.get("instruction", "")


def _io_text(task_id: str, model: str) -> str:
    p = EVAL / CONFIGS[model] / "tasks" / task_id / "logs" / "environment_io.md"
    return p.read_text()


def _passed(task_id: str, model: str) -> bool | None:
    p = EVAL / CONFIGS[model] / "tasks" / task_id / "evaluation" / "report.md"
    return passed_from_report(p.read_text()) if p.exists() else None


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for feat in FEATURED:
        tid, app = feat["task_id"], feat["app"]
        instruction = _instruction(tid)
        entry = {"task_id": tid, "app": app, "instruction": instruction}
        for model in ("vanilla", "serpo"):
            records = build_step_records(_io_text(tid, model))
            passed = _passed(tid, model)
            doc = {"task_id": tid, "model": model, "app": app,
                   "instruction": instruction, "passed": passed,
                   "num_steps": len(records), "steps": records}
            (out_dir / f"{tid}.{model}.json").write_text(json.dumps(doc, indent=1))
            entry[f"{model}_passed"] = passed
            entry[f"{model}_num_steps"] = len(records)
        manifest.append(entry)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"Wrote {len(manifest)} tasks to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    args = ap.parse_args()
    main(Path(args.out))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/minjeong/Autonomous_agent && python -m pytest viz/tests/test_build.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Generate the real data and sanity-check it**

Run: `cd /data/minjeong/Autonomous_agent && python -m viz.build_trajectory_json`
Expected: `Wrote 2 tasks to .../viz/data`

Run: `cd /data/minjeong/Autonomous_agent && python -c "import json; m=json.load(open('viz/data/manifest.json')); print(m)"`
Expected: two entries; for `552869a_2`: `vanilla_passed=False, serpo_passed=True`, `vanilla_num_steps` large (~50), `serpo_num_steps` 8. For `31dc501_2`: `vanilla_passed=False, serpo_passed=True`.

**If `vanilla_passed`/`serpo_passed` do not match (False→True), STOP** — the report path or task id is wrong; re-verify against `appworld/experiments/outputs/eval/dev/<config>/tasks/<id>/evaluation/report.md` before proceeding.

- [ ] **Step 6: Commit (including generated data)**

```bash
cd /data/minjeong/Autonomous_agent
git add viz/build_trajectory_json.py viz/tests/test_build.py viz/data/
git commit -m "feat(viz): build script + generated trajectory JSON for 2 tasks"
```

---

## Task 5: Player HTML skeleton (start screen + stage markup)

**Files:**
- Create: `viz/player.html`

No tests (static markup; verified visually in Task 7). This task creates structure only;
styling and behavior come in Task 6.

- [ ] **Step 1: Create `viz/player.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AppWorld — Vanilla vs SeRPO</title>
  <link rel="stylesheet" href="player.css">
</head>
<body>
  <!-- Start screen: task selector. Populated from manifest.json by player.js. -->
  <section id="selector">
    <h1>AppWorld Agent — Vanilla GRPO vs SeRPO</h1>
    <p class="sub">Pick a task. Both models play side-by-side: Vanilla fails, SeRPO succeeds.</p>
    <div id="task-cards"></div>
  </section>

  <!-- Stage: hidden until a task is chosen. Two model panels share one clock. -->
  <section id="stage" hidden>
    <div id="controls">
      <button id="back">← tasks</button>
      <button id="playpause">⏸ Pause</button>
      <button id="restart">⟲ Restart</button>
      <label>Speed
        <select id="speed">
          <option value="2000">0.5×</option>
          <option value="1000" selected>1×</option>
          <option value="500">2×</option>
          <option value="250">4×</option>
        </select>
      </label>
      <span id="task-instruction"></span>
    </div>
    <div id="panels">
      <div class="panel vanilla">
        <div class="panel-head">✗ Vanilla GRPO</div>
        <div class="panel-body">
          <div class="chat" data-role="chat"></div>
          <div class="appui" data-role="appui"></div>
        </div>
      </div>
      <div class="panel serpo">
        <div class="panel-head">✓ SeRPO</div>
        <div class="panel-body">
          <div class="chat" data-role="chat"></div>
          <div class="appui" data-role="appui"></div>
        </div>
      </div>
    </div>
  </section>

  <script src="player.js"></script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
cd /data/minjeong/Autonomous_agent
git add viz/player.html
git commit -m "feat(viz): player HTML skeleton (selector + stage)"
```

---

## Task 6: Player CSS + JS (selector, timeline, rendering)

**Files:**
- Create: `viz/player.css`
- Create: `viz/player.js`

- [ ] **Step 1: Create `viz/player.css`**

```css
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, Segoe UI, Roboto, sans-serif;
       background: #eef1f5; color: #1f2937; }

/* ---- Selector ---- */
#selector { max-width: 900px; margin: 8vh auto; text-align: center; padding: 0 20px; }
#selector h1 { font-size: 28px; margin-bottom: 8px; }
#selector .sub { color: #6b7280; margin-bottom: 32px; }
#task-cards { display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; }
.task-card { width: 360px; background: #fff; border-radius: 14px; padding: 20px;
             box-shadow: 0 2px 10px rgba(0,0,0,.08); cursor: pointer; text-align: left;
             transition: transform .12s, box-shadow .12s; }
.task-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,.14); }
.task-card .app { font-size: 12px; text-transform: uppercase; letter-spacing: .05em;
                  color: #6b7280; }
.task-card .instr { font-size: 16px; font-weight: 600; margin: 8px 0 14px; }
.task-card .badge { font-size: 13px; font-weight: 600; }
.task-card .badge .fail { color: #ef4444; }
.task-card .badge .pass { color: #16a34a; }

/* ---- Stage ---- */
#controls { display: flex; align-items: center; gap: 12px; padding: 10px 16px;
            background: #fff; border-bottom: 1px solid #e5e7eb; }
#controls button, #controls select { font-size: 14px; padding: 4px 10px; cursor: pointer; }
#task-instruction { margin-left: auto; color: #374151; font-weight: 600; }
#panels { display: flex; gap: 12px; padding: 12px; height: calc(100vh - 56px); }
.panel { flex: 1; display: flex; flex-direction: column; background: #fff;
         border-radius: 12px; overflow: hidden; border: 3px solid #e5e7eb; }
.panel.vanilla { border-color: #ef4444; }
.panel.serpo { border-color: #22c55e; }
.panel-head { padding: 8px 14px; font-weight: 700; color: #fff; }
.panel.vanilla .panel-head { background: #ef4444; }
.panel.serpo .panel-head { background: #22c55e; }
.panel-body { flex: 1; display: flex; min-height: 0; }
.chat { flex: 1.1; overflow-y: auto; padding: 12px; background: #f6f8fa; }
.appui { flex: 1; overflow-y: auto; padding: 12px; background: #fbfbfd;
         border-left: 1px solid #eceef1; }

/* ---- Chat bubbles ---- */
.bubble { max-width: 92%; margin: 6px 0; padding: 8px 11px; border-radius: 12px;
          font-size: 13px; line-height: 1.45; white-space: pre-wrap;
          animation: pop .18s ease-out; }
.bubble .who { font-size: 11px; font-weight: 700; opacity: .75; margin-bottom: 3px; }
.bubble.agent { background: #dbeafe; color: #1e3a5f; margin-right: auto; }
.bubble.env { background: #e5e7eb; color: #374151; margin-left: auto; }
.bubble code, .bubble .mono { font-family: ui-monospace, Menlo, Consolas, monospace; }
.bubble.env .out { font-family: ui-monospace, Menlo, Consolas, monospace;
                   max-height: 7.5em; overflow: hidden; }
@keyframes pop { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; } }

/* ---- App UI ---- */
.appui h3 { margin: 0 0 10px; font-size: 14px; }
.app-row { padding: 8px 10px; border-radius: 8px; background: #f3f4f6; margin: 6px 0;
           font-size: 13px; display: flex; justify-content: space-between; gap: 8px; }
.app-row.hit { background: #fef9c3; outline: 1px solid #facc15; }
.app-row.changed { background: #dcfce7; outline: 1px solid #22c55e;
                   animation: flash .6s ease-out; }
@keyframes flash { from { background: #bbf7d0; } to { background: #dcfce7; } }
.result-card { padding: 14px; border-radius: 10px; background: #dcfce7; font-weight: 700; }
.result-card.bad { background: #fee2e2; }
.docs-note { color: #6b7280; font-style: italic; }
.done-banner { margin-top: 10px; padding: 6px 10px; border-radius: 8px; font-weight: 700; }
.done-banner.pass { background: #dcfce7; color: #166534; }
.done-banner.fail { background: #fee2e2; color: #991b1b; }
```

- [ ] **Step 2: Create `viz/player.js`**

```javascript
// AppWorld trajectory comparison player.
// Loads manifest.json + per-(task,model) JSON, shows a task selector,
// and animates both models on one shared step clock.

let manifest = [];
let timer = null;
let paused = false;
let stepIndex = 0;
let current = null; // { vanilla: doc, serpo: doc, maxSteps }

async function loadJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
  return r.json();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

// ---- Selector ----
async function initSelector() {
  manifest = await loadJSON('data/manifest.json');
  const wrap = document.getElementById('task-cards');
  wrap.innerHTML = '';
  for (const t of manifest) {
    const card = document.createElement('div');
    card.className = 'task-card';
    card.innerHTML = `
      <div class="app">${escapeHtml(t.app)}</div>
      <div class="instr">${escapeHtml(t.instruction)}</div>
      <div class="badge">Vanilla <span class="fail">✗ fail</span>
        &nbsp;→&nbsp; SeRPO <span class="pass">✓ pass</span></div>`;
    card.onclick = () => startTask(t.task_id);
    wrap.appendChild(card);
  }
}

// ---- Start a task ----
async function startTask(taskId) {
  const vanilla = await loadJSON(`data/${taskId}.vanilla.json`);
  const serpo = await loadJSON(`data/${taskId}.serpo.json`);
  current = { vanilla, serpo, maxSteps: Math.max(vanilla.num_steps, serpo.num_steps) };

  document.getElementById('selector').hidden = true;
  document.getElementById('stage').hidden = false;
  document.getElementById('task-instruction').textContent = vanilla.instruction;
  for (const p of document.querySelectorAll('.panel')) {
    p.querySelector('[data-role=chat]').innerHTML = '';
    p.querySelector('[data-role=appui]').innerHTML = '';
  }
  stepIndex = 0;
  paused = false;
  document.getElementById('playpause').textContent = '⏸ Pause';
  scheduleNext();
}

function panelFor(model) {
  return document.querySelector(`.panel.${model}`);
}

function renderStep(model, doc) {
  if (stepIndex >= doc.num_steps) {
    maybeDoneBanner(model, doc);
    return;
  }
  const step = doc.steps[stepIndex];
  const panel = panelFor(model);
  const chat = panel.querySelector('[data-role=chat]');

  const agent = document.createElement('div');
  agent.className = 'bubble agent';
  agent.innerHTML = `<div class="who">🤖 Agent — step ${step.step}</div>` +
                    `<code class="mono">${escapeHtml(step.code)}</code>`;
  chat.appendChild(agent);

  const env = document.createElement('div');
  env.className = 'bubble env';
  env.innerHTML = `<div class="who">🌐 AppWorld</div>` +
                  `<div class="out">${escapeHtml(step.output)}</div>`;
  chat.appendChild(env);
  chat.scrollTop = chat.scrollHeight;

  renderAppUI(model, step.ui_state);
}

function renderAppUI(model, ui) {
  const appui = panelFor(model).querySelector('[data-role=appui]');
  if (!ui) return;
  let html = `<h3>${escapeHtml(ui.title || '')}</h3>`;
  if (ui.kind === 'docs') {
    html += `<div class="docs-note">Looking up available APIs…</div>`;
  } else if (ui.kind === 'login') {
    html += `<div class="app-row">🔓 Signed in</div>`;
  } else if (ui.kind === 'venmo_transactions') {
    for (const r of ui.rows) {
      const hit = /electric|power/i.test(r.description || '');
      html += `<div class="app-row ${hit ? 'hit' : ''}">
        <span>${escapeHtml(r.sender)} → ${escapeHtml(r.receiver)}: ${escapeHtml(r.description)}</span>
        <b>$${escapeHtml(r.amount)}</b></div>`;
    }
  } else if (ui.kind === 'phone_alarms') {
    for (const r of ui.rows) {
      html += `<div class="app-row ${r.changed ? 'changed' : ''}">
        <span>⏰ ${escapeHtml(r.label)} (${escapeHtml(r.time)})</span>
        <b>snooze ${escapeHtml(r.snooze_minutes)}m</b></div>`;
    }
  } else if (ui.kind === 'result') {
    html += `<div class="result-card">Task submitted</div>`;
  } else if (ui.kind === 'api_call') {
    html += `<div class="app-row">${escapeHtml(ui.title)}</div>`;
  }
  appui.innerHTML = html;
}

function maybeDoneBanner(model, doc) {
  const appui = panelFor(model).querySelector('[data-role=appui]');
  if (appui.querySelector('.done-banner')) return;
  const banner = document.createElement('div');
  const pass = doc.passed === true;
  banner.className = `done-banner ${pass ? 'pass' : 'fail'}`;
  banner.textContent = pass ? '✓ Task passed' : '✗ Task failed';
  appui.appendChild(banner);
}

function tick() {
  renderStep('vanilla', current.vanilla);
  renderStep('serpo', current.serpo);
  stepIndex++;
  if (stepIndex >= current.maxSteps) {
    // Final render to flush done banners on both sides.
    renderStep('vanilla', current.vanilla);
    renderStep('serpo', current.serpo);
    timer = null;
    return;
  }
  scheduleNext();
}

function scheduleNext() {
  if (paused) return;
  const delay = parseInt(document.getElementById('speed').value, 10);
  timer = setTimeout(tick, delay);
}

// ---- Controls ----
function wireControls() {
  document.getElementById('back').onclick = () => {
    if (timer) clearTimeout(timer);
    timer = null;
    document.getElementById('stage').hidden = true;
    document.getElementById('selector').hidden = false;
  };
  document.getElementById('playpause').onclick = (e) => {
    paused = !paused;
    e.target.textContent = paused ? '▶ Play' : '⏸ Pause';
    if (!paused && !timer) scheduleNext();
  };
  document.getElementById('restart').onclick = () => {
    if (current) startTask(current.vanilla.task_id);
  };
}

wireControls();
initSelector().catch(err => {
  document.getElementById('task-cards').textContent = 'Failed to load data: ' + err.message;
});
```

- [ ] **Step 3: Commit**

```bash
cd /data/minjeong/Autonomous_agent
git add viz/player.css viz/player.js
git commit -m "feat(viz): player styling + selector/timeline animation"
```

---

## Task 7: Manual verification + README

**Files:**
- Create: `viz/README.md`

The player uses `fetch()`, which needs an HTTP server (file:// blocks fetch). Document this.

- [ ] **Step 1: Serve and open the player**

Run: `cd /data/minjeong/Autonomous_agent/viz && python -m http.server 8123`
Then open `http://localhost:8123/player.html` in a browser.

- [ ] **Step 2: Verify behavior (manual checklist)**

Confirm:
- Start screen shows **two** task cards (Venmo "How much have I paid in electricity bill…", Phone "Set my weekend wake up alarm snooze…").
- Clicking the Venmo card hides the selector and starts playback; both panels fill in.
- Vanilla panel keeps streaming steps after SeRPO has finished (SeRPO shows "✓ Task passed" while Vanilla is still going, then ends "✗ Task failed").
- Right-hand app UI: Venmo shows transaction rows with electricity rows highlighted; ends on a result.
- "← tasks" returns to the selector; clicking the Phone card plays the alarms task (alarm row flashes green when snooze updates to 15m).
- Pause/Play, Restart, and Speed all work.
- No errors in the browser console.

**If any check fails, STOP and fix the relevant component before claiming completion.** Take a screenshot of the running Venmo comparison for the record.

- [ ] **Step 3: Write `viz/README.md`**

```markdown
# Trajectory Comparison Video

Side-by-side player: Vanilla-GRPO (fails) vs SeRPO (succeeds) on the same AppWorld task.
Left = chat bubbles (agent code + AppWorld response). Right = mock app UI reflecting the action.

## Regenerate data
    python -m viz.build_trajectory_json
Writes `viz/data/*.json` from the eval outputs under
`appworld/experiments/outputs/eval/dev/{vanilla60,serpoavg60}_test_test_normal/`.

## Run the player
    cd viz && python -m http.server 8123
    # open http://localhost:8123/player.html
`fetch()` requires HTTP — opening the file directly (file://) will not load the data.

## Record a video
Pick a task on the start screen, set the speed, and screen-record the panel area.
The control bar is above the panels and can be cropped out.

## Featured tasks
- `552869a_2` (venmo) — "How much have I paid in electricity bill on venmo this year so far?"
- `31dc501_2` (phone) — "Set my weekend wake up alarm snooze to 15 minutes."

Both are confirmed flip tasks (Vanilla fails the unit test, SeRPO passes) from the matched
`test_normal` eval runs. This is a per-task qualitative illustration, not an aggregate
win-rate claim.

## Tests
    python -m pytest viz/tests -v
```

- [ ] **Step 4: Run the full test suite**

Run: `cd /data/minjeong/Autonomous_agent && python -m pytest viz/tests -v`
Expected: PASS (all tests from Tasks 1–4).

- [ ] **Step 5: Commit**

```bash
cd /data/minjeong/Autonomous_agent
git add viz/README.md
git commit -m "docs(viz): README + manual verification complete"
```

---

## Self-Review Notes

- **Spec coverage:** extractor (Task 1+4), per-app deriver venmo+phone+fallback (Tasks 2–3),
  player side-by-side with chat+mock-UI (Tasks 5–6), task-selector start screen with
  click-to-start (Task 6 `initSelector`/`startTask`), per-task fail→succeed framing (data
  verified in Task 4 Step 5; banners in Task 6), screen-record workflow (Task 7 README). All
  spec sections map to a task.
- **Type consistency:** `Step` fields (`step, code, output, app, api`) used identically in
  Tasks 1/4. `ui_state` `kind` values (`docs, login, venmo_transactions, phone_alarms,
  result, api_call, idle`) produced in Tasks 2–3 and all handled in `renderAppUI` (Task 6).
  `derive_ui_state(app, api, output, prev)` signature identical across Tasks 2/3/4.
  `passed_from_report` / `build_step_records` names match between Task 4 impl and test.
- **No placeholders:** every code/markup step has complete content.
- **Note:** the `value > 0.0` default in real `show_transactions` output is data, not code —
  the deriver only reads list-of-dicts responses, which is what the real trajectory produces.
```
