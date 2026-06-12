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
The control bar is above the panels and can be cropped out. Both models play on one shared
clock, so SeRPO finishes and shows "✓ Task passed" while Vanilla is still running toward
"✗ Task failed". The long Vanilla venmo run collapses its ~44 blank-code steps into a single
"… N empty steps" note so the chat stays readable.

## Featured tasks
- `552869a_2` (venmo) — "How much have I paid in electricity bill on venmo this year so far?"
- `31dc501_2` (phone) — "Set my weekend wake up alarm snooze to 15 minutes."

Both are confirmed flip tasks (Vanilla fails the unit test, SeRPO passes) from the matched
`test_normal` eval runs. This is a per-task qualitative illustration, not an aggregate
win-rate claim.

## Tests
    python -m pytest viz/tests -v
The player itself (HTML/CSS/JS) has no unit tests; verify it visually in a browser.
