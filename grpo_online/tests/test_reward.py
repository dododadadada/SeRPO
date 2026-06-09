import numpy as np
from grpo_online.reward import score_round

class _FakeTok:
    pad_token_id = 0

def test_score_round_sets_segment_advantages(monkeypatch):
    def fake_tok(path, tokenizer):
        return {"input_ids": [1,2,3,4], "attention_mask": [1,1,1,1],
                "response_mask": [0,1,1,0], "step_token_ranges": [(1,3)]}
    def fake_judge(path):
        return [{"contribution": 5 if "a" in str(path) else 1,
                 "start_step": 1, "end_step": 1}]
    monkeypatch.setattr("grpo_online.reward.tokenize_trajectory", fake_tok)
    items = [{"task_id": "t1", "seed": 1, "lm_calls_path": "a/lm_calls.jsonl", "outcome": 1.0},
             {"task_id": "t1", "seed": 2, "lm_calls_path": "b/rollout.jsonl", "outcome": 0.0}]
    rds = score_round(items, _FakeTok(), fake_judge)
    assert len(rds) == 2
    a0 = np.asarray(rds[0].token_adv); a1 = np.asarray(rds[1].token_adv)
    assert a0[1] > a1[1]
