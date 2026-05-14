"""Extract strict 0/8 fail-only task_id set from Round-0 evaluations."""
from __future__ import annotations

import argparse
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
            f"expected 8 seeds under {rollout_round_dir}, "
            f"found {len(seed_dirs)}: {seed_dirs}"
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
