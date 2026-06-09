"""Build per-method parquet datasets from Round-0 rollouts + joint reward output.

CLI:
  python -m grpo.preprocess.build_dataset \
      --method {vanilla,serpo} \
      --condition {failonly,full} \
      --outcome-type {binary,continuous} \
      --rollout-dir appworld/experiments/outputs/rollout/round0 \
      --joint-dir rubric_reward/results/rollout \
      --output-dir grpo/data/round0

Output filename: ``{method}_{condition}_{outcome_type}.parquet``.

Conditions:
  - ``failonly``: only task_ids where binary success == 0 across all 8 seeds (81 tasks).
  - ``full``: all 90 tasks.

Outcome types (only relevant for vanilla method):
  - ``binary``: 1.0 if all tests pass else 0.0 (AppWorld's ``success`` field).
  - ``continuous``: ``len(passes) / num_tests`` ∈ [0, 1].
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
from grpo.preprocess.make_advantages import (
    RolloutData,
    compute_gigpo_advantage,
    compute_serpo_advantage,
    compute_serpo_avg_advantage,
    compute_vanilla_advantage,
)
from grpo.preprocess.tokenize_trajectory import tokenize_trajectory

logger = logging.getLogger(__name__)

MAX_TOKENS = 32000
MAX_STEPS = 50  # max lm_calls.jsonl lines; matches Phase 2 joint reward filter
MAX_ENV_IO_BYTES = 200_000  # 200KB; matches Phase 2 joint reward filter


def _is_trajectory_too_large(
    lm_calls_path: Path,
    max_steps: int = MAX_STEPS,
    max_env_io_bytes: int = MAX_ENV_IO_BYTES,
) -> tuple[bool, str | None]:
    """Cheap pre-tokenization filter to avoid memory blowups on outlier
    trajectories (e.g., agent infinite loops with huge env_io). Returns
    (too_large, reason)."""
    # Step count: number of non-empty lines in lm_calls.jsonl.
    try:
        with lm_calls_path.open() as f:
            n_steps = sum(1 for line in f if line.strip())
    except Exception as e:
        return True, f"lm_calls_read_error:{e}"
    if n_steps > max_steps:
        return True, f"steps>{max_steps}:{n_steps}"
    # env_io size: located at logs/environment_io.md (sibling of lm_calls.jsonl).
    env_io = lm_calls_path.parent / "environment_io.md"
    if env_io.exists():
        size = env_io.stat().st_size
        if size > max_env_io_bytes:
            return True, f"env_io>{max_env_io_bytes}:{size}"
    return False, None


def _load_outcomes(
    rollout_dir: Path, seed: int, outcome_type: str = "binary",
) -> dict[str, float]:
    """Return ``{task_id: outcome}`` for one seed.

    outcome_type='binary': 1.0 if AppWorld success == True else 0.0.
    outcome_type='continuous': passes / num_tests (clipped to [0, 1]).
    """
    eval_path = rollout_dir / f"seed_{seed}" / "evaluations" / "train.json"
    data = json.loads(eval_path.read_text())
    out: dict[str, float] = {}
    for tid, rec in data["individual"].items():
        if outcome_type == "binary":
            out[tid] = 1.0 if rec.get("success", False) else 0.0
        elif outcome_type == "continuous":
            npass = len(rec.get("passes", []))
            ntest = int(rec.get("num_tests", 0))
            out[tid] = (npass / ntest) if ntest > 0 else 0.0
        else:
            raise ValueError(f"unknown outcome_type: {outcome_type!r}")
    return out


def _load_joint_records(joint_dir: Path, seed: int) -> dict[str, dict[str, Any]]:
    """Load seed_N.jsonl into {task_id: record}."""
    path = joint_dir / f"seed_{seed}.jsonl"
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
    gigpo_kwargs: dict | None = None,
) -> int:
    """Process the rollouts of one task group, compute the method's advantage,
    append rows to a parquet file. Returns number of rollouts written, or 0
    when the group is dropped (e.g., a member exceeds MAX_TOKENS).

    Each input dict must have keys: task_id, seed, lm_calls_path, joint_record, outcome.
    """
    rollouts: list[RolloutData] = []
    for inp in inputs:
        try:
            tok_result = tokenize_trajectory(inp["lm_calls_path"], tokenizer)
        except Exception as e:
            logger.warning(
                "tokenize failed for task=%s seed=%s: %s",
                inp["task_id"], inp["seed"], e,
            )
            return 0
        if len(tok_result["input_ids"]) > MAX_TOKENS:
            logger.warning(
                "len > %d for task=%s seed=%s — dropping whole group",
                MAX_TOKENS, inp["task_id"], inp["seed"],
            )
            return 0
        joint = inp["joint_record"]
        rollouts.append(
            RolloutData(
                task_id=inp["task_id"],
                seed=inp["seed"],
                input_ids=tok_result["input_ids"],
                attention_mask=tok_result["attention_mask"],
                response_mask=tok_result["response_mask"],
                step_token_ranges=tok_result["step_token_ranges"],
                segments=joint.get("segments", []),
                step_anchor_obs=tok_result.get("step_anchor_obs", []),
                outcome=float(inp["outcome"]),
            )
        )

    if method == "vanilla":
        compute_vanilla_advantage(rollouts)
    elif method == "serpo":
        compute_serpo_advantage(rollouts)
    elif method == "serpo_avg":
        compute_serpo_avg_advantage(rollouts)
    elif method == "gigpo":
        compute_gigpo_advantage(rollouts, **(gigpo_kwargs or {}))
    else:
        raise ValueError(f"unknown method: {method}")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(
            [
                {
                    "task_id": r.task_id,
                    "seed": r.seed,
                    "input_ids": r.input_ids,
                    "attention_mask": r.attention_mask,
                    "response_mask": r.response_mask,
                    "advantages": [float(x) for x in r.token_adv.tolist()],
                    "outcome": float(r.outcome),
                    "num_steps": len(r.step_token_ranges),
                }
                for r in rollouts
            ]
        )
        if out_path.exists():
            existing = pq.read_table(out_path)
            table = pa.concat_tables([existing, table])
        pq.write_table(table, out_path)

    return len(rollouts)


def run_build(
    method: str,
    condition: str,
    rollout_dir: Path,
    joint_dir: Path,
    output_dir: Path,
    outcome_type: str = "binary",
    tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct",
    gigpo_kwargs: dict | None = None,
) -> None:
    if condition not in ("failonly", "full"):
        raise ValueError(f"unknown condition: {condition!r}")
    if method == "gigpo" and condition == "failonly":
        raise ValueError(
            "method=gigpo with condition=failonly is degenerate: binary reward "
            "is 0 for every fail-only rollout, so A^E and A^S are all zero. "
            "Use condition=full for GiGPO."
        )
    if outcome_type not in ("binary", "continuous"):
        raise ValueError(f"unknown outcome_type: {outcome_type!r}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    seed_outcomes: dict[int, dict[str, float]] = {}
    seed_joints: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in range(1, 9):
        seed_outcomes[seed] = _load_outcomes(rollout_dir, seed, outcome_type)
        seed_joints[seed] = _load_joint_records(joint_dir, seed)

    # Determine task_id pool based on condition
    if condition == "failonly":
        task_id_pool: list[str] = build_failset(rollout_dir)
        (output_dir / "failonly_task_ids.json").write_text(
            json.dumps(task_id_pool, indent=2)
        )
        logger.info("fail-only task count: %d", len(task_id_pool))
    else:  # full
        # Use union of task_ids across seeds (in practice all seeds share the same set)
        all_tids: set[str] = set()
        for seed_n in range(1, 9):
            all_tids.update(seed_outcomes[seed_n].keys())
        task_id_pool = sorted(all_tids)
        (output_dir / "full_task_ids.json").write_text(
            json.dumps(task_id_pool, indent=2)
        )
        logger.info("full task count: %d", len(task_id_pool))

    out_path = output_dir / f"{method}_{condition}_{outcome_type}.parquet"
    if out_path.exists():
        out_path.unlink()  # overwrite

    n_groups = 0
    n_rows = 0
    n_skipped_groups = 0
    skipped_log_path = output_dir / f"skipped_{method}_{condition}_{outcome_type}.jsonl"
    total_tasks = len(task_id_pool)
    with skipped_log_path.open("w") as skip_log:
        for task_idx, task_id in enumerate(task_id_pool, start=1):
            inputs: list[dict[str, Any]] = []
            group_ok = True
            for seed in range(1, 9):
                lm_calls = (
                    rollout_dir
                    / f"seed_{seed}"
                    / "tasks"
                    / task_id
                    / "logs"
                    / "lm_calls.jsonl"
                )
                if not lm_calls.exists():
                    skip_log.write(
                        json.dumps(
                            {
                                "task_id": task_id,
                                "seed": seed,
                                "reason": "lm_calls_missing",
                            }
                        )
                        + "\n"
                    )
                    group_ok = False
                    break
                # Pre-tokenize size filter to avoid OOM on outlier trajectories.
                too_large, reason = _is_trajectory_too_large(lm_calls)
                if too_large:
                    skip_log.write(
                        json.dumps(
                            {
                                "task_id": task_id,
                                "seed": seed,
                                "reason": reason,
                            }
                        )
                        + "\n"
                    )
                    group_ok = False
                    break
                joint_rec = seed_joints[seed].get(task_id)
                if joint_rec is None and method in ("serpo", "serpo_avg"):
                    skip_log.write(
                        json.dumps(
                            {
                                "task_id": task_id,
                                "seed": seed,
                                "reason": "joint_missing",
                            }
                        )
                        + "\n"
                    )
                    group_ok = False
                    break
                inputs.append(
                    {
                        "task_id": task_id,
                        "seed": seed,
                        "lm_calls_path": lm_calls,
                        "joint_record": joint_rec or {"segments": []},
                        "outcome": float(seed_outcomes[seed].get(task_id, 0.0)),
                    }
                )
            if not group_ok:
                n_skipped_groups += 1
                if task_idx % 10 == 0 or task_idx == total_tasks:
                    logger.info(
                        "progress %d/%d  groups=%d  rows=%d  skipped=%d  (latest skip: %s)",
                        task_idx, total_tasks, n_groups, n_rows, n_skipped_groups, task_id,
                    )
                continue

            written = build_dataset_for_task_group(
                inputs, tokenizer=tokenizer, method=method, out_path=out_path,
                gigpo_kwargs=gigpo_kwargs,
            )
            if written == 0:
                n_skipped_groups += 1
                skip_log.write(
                    json.dumps(
                        {"task_id": task_id, "reason": "tokenize_or_length"}
                    )
                    + "\n"
                )
            else:
                n_groups += 1
                n_rows += written
            if task_idx % 10 == 0 or task_idx == total_tasks:
                logger.info(
                    "progress %d/%d  groups=%d  rows=%d  skipped=%d",
                    task_idx, total_tasks, n_groups, n_rows, n_skipped_groups,
                )

    logger.info(
        "done: %d groups written (%d rows), %d groups skipped — parquet: %s",
        n_groups, n_rows, n_skipped_groups, out_path,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["vanilla", "serpo", "serpo_avg", "gigpo"], required=True)
    ap.add_argument("--condition", choices=["failonly", "full"], default="failonly")
    ap.add_argument(
        "--outcome-type",
        choices=["binary", "continuous"],
        default="binary",
        help="binary: success bool (1/0). continuous: passes/num_tests ∈ [0,1].",
    )
    ap.add_argument(
        "--rollout-dir",
        type=Path,
        default=Path("appworld/experiments/outputs/rollout/round0"),
    )
    ap.add_argument(
        "--joint-dir",
        type=Path,
        default=Path("rubric_reward/results/rollout"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("grpo/data/round0"),
    )
    ap.add_argument(
        "--tokenizer-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF tokenizer for input_ids; must match the policy model being trained.",
    )
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
    args = ap.parse_args()
    gigpo_kwargs = {
        "gamma": args.gamma, "omega": args.omega,
        "norm_mode": args.gigpo_norm_mode,
        "enable_similarity": args.enable_similarity,
        "similarity_thresh": args.similarity_thresh,
    } if args.method == "gigpo" else None
    run_build(
        args.method, args.condition, args.rollout_dir, args.joint_dir,
        args.output_dir, outcome_type=args.outcome_type,
        tokenizer_name=args.tokenizer_name,
        gigpo_kwargs=gigpo_kwargs,
    )


if __name__ == "__main__":
    main()
