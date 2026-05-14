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
from grpo.preprocess.make_advantages import (
    RolloutData,
    compute_serpo_advantage,
    compute_vanilla_advantage,
)
from grpo.preprocess.tokenize_trajectory import tokenize_trajectory

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
                outcome=int(inp["outcome"]),
            )
        )

    if method == "vanilla":
        compute_vanilla_advantage(rollouts)
    elif method == "serpo":
        compute_serpo_advantage(rollouts)
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
                    "outcome": r.outcome,
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
    tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct",
) -> None:
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
                joint_rec = seed_joints[seed].get(task_id)
                if joint_rec is None and method == "serpo":
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
                        "outcome": int(seed_outcomes[seed].get(task_id, False)),
                    }
                )
            if not group_ok:
                n_skipped_groups += 1
                continue

            written = build_dataset_for_task_group(
                inputs, tokenizer=tokenizer, method=method, out_path=out_path,
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
    ap.add_argument("--method", choices=["vanilla", "serpo"], required=True)
    ap.add_argument("--condition", choices=["failonly"], default="failonly")
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
    args = ap.parse_args()
    run_build(
        args.method, args.condition, args.rollout_dir, args.joint_dir, args.output_dir,
    )


if __name__ == "__main__":
    main()
