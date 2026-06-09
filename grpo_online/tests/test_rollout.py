import random
from grpo_online.rollout import sample_tasks, items_from_outputs

def test_sample_tasks_no_repeat_within_round():
    ts = [f"t{i}" for i in range(20)]
    got = sample_tasks(ts, 6, random.Random(0))
    assert len(got) == 6 and len(set(got)) == 6 and set(got) <= set(ts)

def test_items_from_outputs(tmp_path):
    seed = tmp_path / "seed_1"
    (seed / "tasks" / "t1" / "logs").mkdir(parents=True)
    (seed / "tasks" / "t1" / "logs" / "lm_calls.jsonl").write_text("{}\n")
    (seed / "evaluations").mkdir()
    (seed / "evaluations" / "train.json").write_text(
        '{"individual": {"t1": {"success": true, "passes": [1,2], "num_tests": 2}}}')
    items = items_from_outputs(str(tmp_path), seeds=[1], tasks=["t1"], outcome_type="continuous")
    assert items == [{"task_id": "t1", "seed": 1,
                      "lm_calls_path": str(seed / "tasks" / "t1" / "logs" / "lm_calls.jsonl"),
                      "outcome": 1.0}]
