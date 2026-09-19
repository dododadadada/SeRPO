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


def _fake_appworld(tmp_path):
    """A stand-in `appworld` binary: writes the output tree `appworld run` would,
    recording (experiment, ROLLOUT_SEED, --num-processes) so tests can inspect it."""
    import os, stat, textwrap
    bin_ = tmp_path / "fake_appworld_bin"
    bin_.write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        # $1=run $2=experiment ; rest are flags
        exp="$2"; ds="$ROLLOUT_DATASET"
        echo "$exp $ROLLOUT_SEED $4 $VLLM_PORT $MODEL_SERVER_URL" >> "$PWD/calls.log"
        sleep 0.2
        out="$PWD/experiments/outputs/$exp"
        for t in $(cat "$PWD/data/datasets/$ds.txt"); do
          mkdir -p "$out/tasks/$t/logs"; echo '{}' > "$out/tasks/$t/logs/lm_calls.jsonl"
        done
        mkdir -p "$out/evaluations"
        python3 - "$out/evaluations/$ds.json" "$PWD/data/datasets/$ds.txt" <<'PY'
        import json, sys
        ts = [l.strip() for l in open(sys.argv[2]) if l.strip()]
        json.dump({"individual": {t: {"success": True, "passes": [1], "num_tests": 2} for t in ts}}, open(sys.argv[1], "w"))
        PY
        """))
    bin_.chmod(bin_.stat().st_mode | stat.S_IEXEC)
    return str(bin_)


def _cfg(tmp_path, **kw):
    d = {"G": 3, "temperature": 1.0, "top_p": 1.0, "gen_ports": [8101],
         "appworld_num_processes": 6, "output_dir": str(tmp_path / "runs"),
         "outcome_type": "continuous", "seed_parallelism": 1}
    d.update(kw)
    return type("C", (), d)()


def test_run_rollout_sequential_uses_base_experiment(monkeypatch, tmp_path):
    from grpo_online.rollout import run_rollout
    monkeypatch.chdir(tmp_path)
    (tmp_path / "appworld").mkdir()
    bin_ = _fake_appworld(tmp_path)
    items = run_rollout(_cfg(tmp_path), 1, ["t1", "t2"], bin_, "rollout/online/qwen35_9b")
    assert sorted((it["task_id"], it["seed"]) for it in items) == [
        ("t1", 1), ("t1", 2), ("t1", 3), ("t2", 1), ("t2", 2), ("t2", 3)]
    assert all(it["outcome"] == 0.5 for it in items)
    calls = (tmp_path / "appworld" / "calls.log").read_text().splitlines()
    assert calls == [f"rollout/online/qwen35_9b {s} 6 8101 http://localhost:8101" for s in (1, 2, 3)]
    assert (tmp_path / "appworld" / "data" / "datasets" / "online_round_1.txt").read_text() == "t1\nt2\n"
    assert not (tmp_path / "appworld" / "experiments" / "configs").exists()  # no stubs


def test_run_rollout_parallel_seeds_use_per_seed_experiment_stubs(monkeypatch, tmp_path):
    from grpo_online.rollout import run_rollout
    monkeypatch.chdir(tmp_path)
    (tmp_path / "appworld").mkdir()
    bin_ = _fake_appworld(tmp_path)
    items = run_rollout(_cfg(tmp_path, seed_parallelism=3), 2, ["t1"], bin_, "rollout/online/qwen35_9b")
    assert sorted(it["seed"] for it in items) == [1, 2, 3]
    for s in (1, 2, 3):
        stub = tmp_path / "appworld" / "experiments" / "configs" / f"rollout/online/qwen35_9b_seed{s}.jsonnet"
        assert stub.read_text() == 'import "qwen35_9b.jsonnet"\n'
        assert (tmp_path / "runs" / "rollouts" / "round_2" / f"seed_{s}" / "tasks" / "t1" / "logs" / "lm_calls.jsonl").exists()
    calls = sorted((tmp_path / "appworld" / "calls.log").read_text().splitlines())
    assert calls == [f"rollout/online/qwen35_9b_seed{s} {s} 6 8101 http://localhost:8101" for s in (1, 2, 3)]
    # nothing left behind in appworld's own output tree
    assert not any((tmp_path / "appworld" / "experiments" / "outputs").rglob("lm_calls.jsonl"))
