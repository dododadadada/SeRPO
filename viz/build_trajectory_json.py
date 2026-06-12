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
        ui = derive_ui_state(s.app, s.api, s.output, prev, s.code)
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
