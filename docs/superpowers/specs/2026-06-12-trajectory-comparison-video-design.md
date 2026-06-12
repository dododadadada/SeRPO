# AppWorld Trajectory Comparison Video — Design

**Date:** 2026-06-12
**Status:** Design (pending implementation plan)
**Author:** brainstormed with Claude

## Goal

Produce a polished, animated video that plays **two AppWorld agent trajectories on the
same task, side-by-side**: a **Vanilla-GRPO** model that **fails** and a **SeRPO** model
that **succeeds**. Each model's view shows:

- **Left:** chat bubbles — the agent's generated Python code, then the AppWorld
  environment's response, streaming step by step.
- **Right:** a **mock app UI** for the app being used, which visually updates to reflect
  the agent's real action (e.g. a Venmo transactions list, a Phone alarms screen).

The video is produced as a **self-contained HTML/CSS/JS page animated on a timeline** and
**screen-recorded** by the user (no headless render pipeline, no extra dependencies).

## Scope and honest framing

This is a **qualitative illustration**, not a win-rate claim. On the matched dev/test_normal
eval runs, SeRPO does **not** beat Vanilla in aggregate pass count (they are roughly tied;
e.g. on test_normal-168, Vanilla 32 vs SeRPO 30). What the video shows is a real,
reproducible phenomenon: **specific tasks where Vanilla fails and SeRPO recovers** ("flip"
tasks). The narrative is "here are cases where SeRPO's segment-level signal recovers a
failure mode Vanilla can't," and the spec, the page, and any caption must not over-claim
beyond that.

## Featured tasks (both are single-app "flip" tasks)

Selected from the matched `vanilla60_test_test_normal` vs `serpoavg60_test_test_normal`
eval runs (the largest matched, held-out pool), confirmed Vanilla-fail / SeRPO-pass:

| Task | App | Vanilla | SeRPO | Why chosen |
|---|---|---|---|---|
| `552869a_2` | **venmo** | 50 steps → FAIL | 8 steps → PASS | Dramatic step-count contrast; clean SeRPO trajectory (login → explore API → query transactions → compute total → answer `144.0`). The centerpiece. |
| `31dc501_2` | **phone** | 8 steps → FAIL | 11 steps → PASS | Alarm-snooze task. Both call `update_alarm`; Vanilla returns a wrong/garbled answer, SeRPO answers correctly. Shows a subtler recovery + a second app. |

Because each task touches exactly one app, we build only **two mock app UIs** (Venmo, Phone).

**Data source paths** (per task, per model):
```
appworld/experiments/outputs/eval/dev/vanilla60_test_test_normal/tasks/<task_id>/logs/environment_io.md
appworld/experiments/outputs/eval/dev/serpoavg60_test_test_normal/tasks/<task_id>/logs/environment_io.md
appworld/experiments/outputs/eval/dev/<config>/tasks/<task_id>/evaluation/report.md   # pass/fail
```
The task instruction is read from the AppWorld task metadata at build time (located by the
extractor; `task.json` path resolved via the appworld dataset, not assumed under the output dir).

## Layout

**Side-by-side, both panels animate together** (chosen over sequential). The two models play
on the same wall-clock timeline so the audience sees Vanilla *still flailing* while SeRPO has
already finished — the 50-vs-8 gap is the point and only lands simultaneously.

```
┌─────────────────────────────┬─────────────────────────────┐
│  ✗ Vanilla GRPO             │  ✓ SeRPO                    │
│  ┌──────────┬────────────┐  │  ┌──────────┬────────────┐  │
│  │ chat     │ mock app   │  │  │ chat     │ mock app   │  │
│  │ bubbles  │ UI (right) │  │  │ bubbles  │ UI (right) │  │
│  │ (left)   │            │  │  │ (left)   │            │  │
│  └──────────┴────────────┘  │  └──────────┴────────────┘  │
└─────────────────────────────┴─────────────────────────────┘
       red header / border            green header / border
```

Chat-bubble styling: **light theme, conversation framing.** Agent bubbles (left-aligned,
"🤖 Agent") contain the generated code in **monospace** so it stays honest that the agent
writes real code. Environment bubbles (right-aligned, "🌐 AppWorld") contain the response,
truncated/folded when very long (e.g. the API-docs dumps) with the full text available but
collapsed by default.

## Components

### 1. Trajectory extractor — `build_trajectory_json.py`
One-time, CPU-only data prep. No GPU.

- **Input:** the four `environment_io.md` files (2 tasks × 2 models) + the two `report.md`
  files for pass/fail + task instructions.
- **Parse:** split each file on `### Environment Interaction N` blocks; for each, extract the
  ` ```python ... ``` ` code block and the following ` ``` ... ``` ` output block into
  `{step, code, output}`.
- **Classify:** for each step, detect the app and API via regex on `apis.<app>.<api>(`.
  Record `{app, api}` per step (a step may call multiple APIs; record the primary/last
  state-changing one for the UI driver, keep all for completeness).
- **Output:** one `trajectory.json` per (task, model), plus a top-level `manifest.json`
  pairing them: `{task_id, instruction, app, vanilla: {...}, serpo: {...}, vanilla_passed,
  serpo_passed}`.

Interface: `python build_trajectory_json.py --out viz/data/` → writes JSON; idempotent.

### 2. App-state deriver (per-app mapping)
The only genuinely hand-crafted part, scoped to **Venmo + Phone** only.

A small mapping from `(app, api)` → a description of what the right-hand mock UI should
render *after* that step, computed from the step's parsed `output`. Examples:

**Venmo:**
- `login` → render logged-in home (account name, balance placeholder).
- `show_transactions(...)` → render the returned transactions as a list of rows
  (sender → receiver, amount, description, date); highlight rows the agent's code filters on
  (e.g. "electricity").
- `complete_task(answer=...)` → render a result card showing the submitted answer
  (e.g. `144.0`), styled green on the SeRPO side, and (on Vanilla) whatever wrong/absent
  answer it submitted.
- API-docs / doc-lookup calls → a neutral "📖 reading API docs" state (no app change).

**Phone:**
- `login` → logged-in home.
- `show_alarms(...)` → render an alarms list (label, time, snooze minutes); highlight the
  weekend alarm being targeted.
- `update_alarm(...)` → animate the targeted alarm's snooze value changing.
- `complete_task(answer=...)` → result card with the submitted answer.

**Fallback:** any `(app, api)` not in the table renders a generic "⚙️ API call: `<api>`"
indicator with the raw response folded — so unmapped steps degrade gracefully rather than
break. The deriver `log()`s which APIs hit the fallback so we know what the video glosses over.

The deriver is pure functions `derive_ui_state(app, api, parsed_output, prev_state) -> ui_state`,
unit-testable independently of the player.

### 3. Player page — `player.html` (self-contained)
- Loads `manifest.json` + the four `trajectory.json` files (inlined or fetched locally).
- Renders the side-by-side layout; each quadrant = chat column + mock-app column.
- **Timeline animation:** a single shared clock advances both models. Per step: agent bubble
  types in → env bubble appears → mock-app UI transitions to the derived state → pause →
  next step. Vanilla's 50 steps and SeRPO's 8 share the clock, so SeRPO finishes early and
  sits on its success card while Vanilla continues.
- **Controls:** play / pause / step-forward / step-back / speed (0.5×–4×) so the user drives
  pacing live while screen-recording. Controls are visually outside the "stage" so they can
  be cropped out of the recording.
- No build step, no framework — plain HTML/CSS/JS so it opens by double-click.

### 4. Output / recording
User opens `player.html`, sets speed, hits play, and screen-records the stage area. No
automated MP4 export in scope (can be added later via Playwright if desired).

## File layout (proposed)
```
viz/
  build_trajectory_json.py     # extractor (component 1)
  app_state.py                 # per-app deriver (component 2), or JS-side — see open note
  player.html                  # player (component 3)
  player.css
  player.js
  data/                        # generated JSON (gitignored or committed small)
    manifest.json
    552869a_2.vanilla.json
    552869a_2.serpo.json
    31dc501_2.vanilla.json
    31dc501_2.serpo.json
  tests/
    test_extractor.py
    test_app_state.py
```

**Open implementation note (not a blocker):** the app-state deriver can live in Python (run
at build time, bake `ui_state` into the JSON) **or** in JS (run in the browser from raw
parsed steps). Baking it at build time keeps `player.js` dumb and makes the deriver
Python-unit-testable — **recommended default**. The plan will confirm.

## Testing
- **Extractor:** unit tests on a fixture `environment_io.md` → assert correct step count,
  code/output split, and `(app, api)` classification.
- **Deriver:** unit tests per mapped API → assert expected `ui_state` from a sample parsed
  output; assert fallback for an unmapped API.
- **Player:** manual visual verification (it's a video tool); a smoke test that the page loads
  the JSON and renders N steps without JS errors.

## Non-goals
- No general engine for arbitrary trajectories/apps (only Venmo + Phone, only these 2 tasks).
- No automated video encoding (screen-record manually).
- No segment/reward overlay (the audience watches the agent act; SeRPO's mechanism is the
  *reason* for the contrast, not shown on screen).
- No pixel-perfect clone of real Venmo/Phone apps — "app-like enough to read the action."
