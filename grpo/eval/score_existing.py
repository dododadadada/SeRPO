"""Score existing trajectories for an appworld experiment, skipping missing tasks.

Usage:
  python -m grpo.eval.score_existing <experiment_name> <dataset_name>

Loads the dataset's task list and calls evaluate_task per task, skipping any
without a tasks/{tid}/dbs directory (i.e., tasks the killed run never reached).
Writes evaluations/{dataset}.json in the same format as appworld's own writer.
"""
from __future__ import annotations

import sys
from pathlib import Path

from appworld.common.io import write_json
from appworld.common.path_store import path_store
from appworld.evaluator import CachedDBHandler, evaluate_task
from appworld.evaluator import Metric
from appworld.task import load_task_ids


def main() -> None:
    experiment_name, dataset_name = sys.argv[1], sys.argv[2]
    all_tids = load_task_ids(dataset_name=dataset_name)
    exp_root = Path(path_store.experiment_outputs) / experiment_name
    metric = Metric()
    CachedDBHandler.reset()
    n_done, n_skipped, n_failed = 0, 0, 0
    for tid in all_tids:
        task_dir = exp_root / "tasks" / tid
        if not (task_dir / "dbs").is_dir():
            n_skipped += 1
            continue
        try:
            tt = evaluate_task(
                task_id=tid,
                experiment_name=experiment_name,
                suppress_errors=True,
                save_report=True,
            )
            metric(tid, tt)
            n_done += 1
        except Exception as e:
            print(f"  failed {tid}: {e}", flush=True)
            n_failed += 1
        finally:
            CachedDBHandler.reset()
    print(f"done={n_done}  skipped={n_skipped}  failed={n_failed}", flush=True)

    eval_dict = metric.get_metrics(include_details=True, reset=True)
    out = exp_root / "evaluations" / f"{dataset_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(eval_dict, str(out), silent=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
