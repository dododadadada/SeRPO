from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("online")


def sample_tasks(all_tasks, M, rng):
    """No-repeat sample of M task ids from all_tasks using rng."""
    pool = list(all_tasks)
    return rng.sample(pool, min(M, len(pool)))


def _eval_individual(seed_dir: Path) -> dict:
    """Load the per-task evaluation dict from a seed's evaluations directory.

    AppWorld writes the evaluation file named after the dataset that was run
    (e.g. ``train.json`` for the train split, ``online_round_3.json`` for a
    custom online-round subset). There is exactly one ``*.json`` per
    ``evaluations/`` directory, so we glob for it rather than hard-coding a name.
    """
    ev_dir = seed_dir / "evaluations"
    ev_files = sorted(ev_dir.glob("*.json")) if ev_dir.exists() else []
    if not ev_files:
        return {}
    return json.loads(ev_files[0].read_text()).get("individual", {})


def items_from_outputs(base, seeds, tasks, outcome_type="continuous"):
    """Read an appworld output tree into flat items.

    For each (seed, task) with an existing ``lm_calls.jsonl`` log, emit
    ``{task_id, seed, lm_calls_path, outcome}``. Continuous outcome is the
    fraction of unit tests passed (passes/num_tests); binary outcome is 1.0 on
    success else 0.0.
    """
    items = []
    skipped = 0
    for s in seeds:
        seed_dir = Path(base) / f"seed_{s}"
        ind = _eval_individual(seed_dir)
        for t in tasks:
            lm = seed_dir / "tasks" / t / "logs" / "lm_calls.jsonl"
            if not lm.exists():
                skipped += 1
                logger.warning(
                    "items_from_outputs: no lm_calls.jsonl for task=%s seed=%s "
                    "(%s) — skipping", t, s, lm)
                continue
            rec = ind.get(t, {})
            if outcome_type == "continuous":
                npass = len(rec.get("passes", []))
                ntest = int(rec.get("num_tests", 0))
                outcome = (npass / ntest) if ntest > 0 else 0.0
            else:
                outcome = 1.0 if rec.get("success") else 0.0
            items.append(
                {
                    "task_id": t,
                    "seed": s,
                    "lm_calls_path": str(lm),
                    "outcome": outcome,
                }
            )
    if skipped:
        logger.warning(
            "items_from_outputs: %d (seed,task) pairs had no lm_calls.jsonl; "
            "%d items collected", skipped, len(items))
    return items


def run_rollout(cfg, round_idx, tasks, appworld_bin, experiment):
    """Roll out the sampled ``tasks`` for G seeds via ``appworld run``.

    Task-subset mechanism (verified against appworld source): ``ROLLOUT_DATASET``
    is resolved by ``appworld.task.load_task_ids`` to a *registered dataset name*
    (validated against ``valid_dataset_names()``, which lists every ``*.txt`` in
    ``appworld/data/datasets/``) and read from
    ``appworld/data/datasets/<name>.txt``. It does NOT accept an arbitrary path.
    So to restrict a round to M task ids we write
    ``appworld/data/datasets/online_round_<i>.txt`` containing only those ids and
    set ``ROLLOUT_DATASET=online_round_<i>``; the name becomes valid as soon as
    the file exists.

    Per seed: appworld writes its output tree to
    ``appworld/experiments/outputs/<experiment>``; we move it to
    ``<output_dir>/rollouts/round_<i>/seed_<s>`` (mirroring run_round0_rollout_9b.sh).
    Returns the parsed items via ``items_from_outputs``.
    """
    base = Path(cfg.output_dir) / "rollouts" / f"round_{round_idx}"
    base.mkdir(parents=True, exist_ok=True)

    appworld_root = Path.cwd() / "appworld"

    # Write the task subset as a registered dataset name.
    ds_name = f"online_round_{round_idx}"
    datasets_dir = appworld_root / "data" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    (datasets_dir / f"{ds_name}.txt").write_text("\n".join(tasks) + "\n")

    port = cfg.gen_ports[0]
    out_dir = appworld_root / "experiments" / "outputs" / experiment

    for s in range(1, cfg.G + 1):
        # Start each seed from a clean experiment output dir.
        if out_dir.exists():
            shutil.rmtree(out_dir)

        env = {
            **os.environ,
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "dummy"),
            "NO_API_KEY": os.environ.get("NO_API_KEY", "dummy"),
            "ROLLOUT_SEED": str(s),
            "ROLLOUT_TEMPERATURE": str(cfg.temperature),
            "ROLLOUT_TOP_P": str(cfg.top_p),
            "ROLLOUT_DATASET": ds_name,
            "VLLM_PORT": str(port),
        }
        subprocess.run(
            [
                appworld_bin,
                "run",
                experiment,
                "--num-processes",
                "4",
                "--with-evaluation",
                "--without-setup",
            ],
            env=env,
            check=False,
            cwd=str(appworld_root),
        )

        dst = base / f"seed_{s}"
        if dst.exists():
            shutil.rmtree(dst)
        if out_dir.exists():
            out_dir.rename(dst)

    outcome_type = getattr(cfg, "outcome_type", "continuous")
    return items_from_outputs(
        str(base), list(range(1, cfg.G + 1)), tasks, outcome_type
    )
