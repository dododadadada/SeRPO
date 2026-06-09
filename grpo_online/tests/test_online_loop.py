from grpo_online.online_loop import run_online, Deps

class FakeTrainer:
    def __init__(self): self.calls = []
    def train_on_batch(self, rds, K): self.calls.append(("train", len(rds), K)); return {"kl_loss": 0.001, "loss": 0.1}
    def save_adapter(self, p): self.calls.append(("save", p))

def test_loop_runs_and_syncs(monkeypatch, tmp_path):
    order = []
    deps = Deps(
        all_tasks=["t1","t2","t3"],
        rollout=lambda rnd, tasks: (order.append(("rollout", rnd)) or [{"task_id": tasks[0], "seed": 1, "lm_calls_path": "x", "outcome": 1.0}]),
        judge=lambda p: [{"contribution": 3, "start_step": 1, "end_step": 1}],
        tokenizer=None, trainer=FakeTrainer(),
        reload_servers=lambda path: order.append(("reload", path)),
        score_round=lambda items, tok, judge: items,
    )
    cfg = type("C", (), {"M": 2, "K": 1, "N_rounds": 2, "adapter_dir": str(tmp_path/"ad"),
                         "kl_halt_threshold": 5.0, "seed": 0})()
    run_online(cfg, deps)
    assert [o for o in order if o[0] in ("rollout","reload")] == [
        ("rollout",1),("reload",str(tmp_path/"ad")),("rollout",2),("reload",str(tmp_path/"ad"))]
    assert ("train", 1, 1) in deps.trainer.calls

def test_empty_round_is_skipped(tmp_path):
    """A round whose score_round returns [] must be skipped: no train_on_batch,
    no save/reload, and the loop continues to the next round without crashing."""
    order = []
    trainer = FakeTrainer()
    # round 1 -> empty (dropped), round 2 -> one item
    def score(items, tok, judge):
        return [] if not getattr(score, "second", False) else items
    def rollout(rnd, tasks):
        if rnd == 2:
            score.second = True
        return [{"task_id": tasks[0], "seed": 1, "lm_calls_path": "x", "outcome": 1.0}]
    deps = Deps(
        all_tasks=["t1", "t2", "t3"],
        rollout=rollout,
        judge=lambda p: [], tokenizer=None, trainer=trainer,
        reload_servers=lambda path: order.append(("reload", path)),
        score_round=score,
    )
    cfg = type("C", (), {"M": 2, "K": 1, "N_rounds": 2, "adapter_dir": str(tmp_path / "ad"),
                         "kl_halt_threshold": 5.0, "seed": 0})()
    done = run_online(cfg, deps)
    assert done == 2
    train_calls = [c for c in trainer.calls if c[0] == "train"]
    assert len(train_calls) == 1  # only round 2 trained; round 1 skipped
    assert [o for o in order if o[0] == "reload"] == [("reload", str(tmp_path / "ad"))]


def test_loop_halts_on_kl(tmp_path):
    class Boom(FakeTrainer):
        def train_on_batch(self, rds, K): return {"kl_loss": 99.0, "loss": 1.0}
    deps = Deps(all_tasks=["t1"], rollout=lambda r,t:[{"task_id":"t1","seed":1,"lm_calls_path":"x","outcome":1.0}],
                judge=lambda p:[], tokenizer=None, trainer=Boom(),
                reload_servers=lambda p: None, score_round=lambda i,t,j: i)
    cfg = type("C", (), {"M":1,"K":1,"N_rounds":5,"adapter_dir":str(tmp_path/"ad"),"kl_halt_threshold":5.0,"seed":0})()
    rounds = run_online(cfg, deps)
    assert rounds == 1
