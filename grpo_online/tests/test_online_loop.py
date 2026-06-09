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

def test_loop_halts_on_kl(tmp_path):
    class Boom(FakeTrainer):
        def train_on_batch(self, rds, K): return {"kl_loss": 99.0, "loss": 1.0}
    deps = Deps(all_tasks=["t1"], rollout=lambda r,t:[{"task_id":"t1","seed":1,"lm_calls_path":"x","outcome":1.0}],
                judge=lambda p:[], tokenizer=None, trainer=Boom(),
                reload_servers=lambda p: None, score_round=lambda i,t,j: i)
    cfg = type("C", (), {"M":1,"K":1,"N_rounds":5,"adapter_dir":str(tmp_path/"ad"),"kl_halt_threshold":5.0,"seed":0})()
    rounds = run_online(cfg, deps)
    assert rounds == 1
