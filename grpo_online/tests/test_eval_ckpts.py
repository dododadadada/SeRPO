from grpo_online.eval_ckpts import tgc_sgc, ckpt_sort_key


def test_tgc_sgc_counts_tasks_and_scenarios():
    ind = {"a_1": {"success": True, "passes": [1, 2], "num_tests": 2},
           "a_2": {"success": False, "passes": [1], "num_tests": 2},
           "b_1": {"success": True, "passes": [1, 2, 3], "num_tests": 3}}
    r = tgc_sgc(ind)
    assert r["n_tasks"] == 3 and r["n_scenarios"] == 2
    assert abs(r["tgc"] - 2 / 3) < 1e-9 and abs(r["sgc"] - 1 / 2) < 1e-9
    assert abs(r["mean_test_pass_frac"] - (1 + 0.5 + 1) / 3) < 1e-9


def test_ckpt_sort_key():
    assert sorted(["ckpt_round_10", "ckpt_round_2", "base"], key=ckpt_sort_key) == ["base", "ckpt_round_2", "ckpt_round_10"]


def test_glob_ckpts_skips_vllm_export_dirs(tmp_path):
    """load_adapter writes <ckpt>_vllm exports beside each checkpoint; a later
    --ckpts all must not evaluate those copies as if they were checkpoints."""
    from grpo_online.eval_ckpts import glob_ckpts
    for n in ("ckpt_round_2", "ckpt_round_2_vllm", "ckpt_round_4", "ckpt_round_4_vllm"):
        (tmp_path / n).mkdir()
    (tmp_path / "adapter_current").mkdir()
    got = sorted(p.rsplit("/", 1)[-1] for p in glob_ckpts(str(tmp_path)))
    assert got == ["ckpt_round_2", "ckpt_round_4"]
