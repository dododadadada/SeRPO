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


def test_score_round_drops_item_on_judge_failure(monkeypatch):
    """A judge_fn that raises for one item drops only that item; others are
    still scored and no exception propagates."""
    def fake_tok(path, tokenizer):
        return {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1],
                "response_mask": [0, 1, 1, 0], "step_token_ranges": [(1, 3)]}

    def flaky_judge(path):
        if "boom" in str(path):
            raise RuntimeError("judge API 500")
        return [{"contribution": 3, "start_step": 1, "end_step": 1}]

    monkeypatch.setattr("grpo_online.reward.tokenize_trajectory", fake_tok)
    items = [
        {"task_id": "t1", "seed": 1, "lm_calls_path": "ok/lm_calls.jsonl", "outcome": 1.0},
        {"task_id": "t1", "seed": 2, "lm_calls_path": "boom/lm_calls.jsonl", "outcome": 0.0},
    ]
    rds = score_round(items, _FakeTok(), flaky_judge)
    assert len(rds) == 1
    assert rds[0].seed == 1


def test_score_round_all_judges_fail_returns_empty(monkeypatch):
    """If every item's judge fails, the whole round yields [] (loop will skip)."""
    def fake_tok(path, tokenizer):
        return {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1],
                "response_mask": [0, 1, 1, 0], "step_token_ranges": [(1, 3)]}

    def boom_judge(path):
        raise RuntimeError("judge down")

    monkeypatch.setattr("grpo_online.reward.tokenize_trajectory", fake_tok)
    items = [{"task_id": "t1", "seed": 1, "lm_calls_path": "x/lm_calls.jsonl", "outcome": 1.0}]
    rds = score_round(items, _FakeTok(), boom_judge)
    assert rds == []


def test_score_round_parallel_matches_sequential_and_skips_bad_items(monkeypatch):
    """max_workers>1: same output as sequential; items dropped by tokenize/length
    are never sent to the judge; a failing judge call drops only its item."""
    import threading
    from grpo_online import reward as R
    def fake_tok(path, tokenizer):
        if "toolong" in str(path):
            return {"input_ids": [1] * (R.MAX_TOKENS + 1), "attention_mask": [1] * (R.MAX_TOKENS + 1),
                    "response_mask": [1] * (R.MAX_TOKENS + 1), "step_token_ranges": [(0, 1)]}
        return {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1],
                "response_mask": [0, 1, 1, 0], "step_token_ranges": [(1, 3)]}
    judged, lock = [], threading.Lock()
    def judge(path):
        with lock: judged.append(str(path))
        if "boom" in str(path): raise RuntimeError("500")
        return [{"contribution": 5 if "hi" in str(path) else 1, "start_step": 1, "end_step": 1}]
    monkeypatch.setattr("grpo_online.reward.tokenize_trajectory", fake_tok)
    items = [{"task_id": "t1", "seed": 1, "lm_calls_path": "hi/lm_calls.jsonl", "outcome": 1.0},
             {"task_id": "t1", "seed": 2, "lm_calls_path": "lo/lm_calls.jsonl", "outcome": 0.0},
             {"task_id": "t1", "seed": 3, "lm_calls_path": "toolong/lm_calls.jsonl", "outcome": 0.0},
             {"task_id": "t1", "seed": 4, "lm_calls_path": "boom/lm_calls.jsonl", "outcome": 0.0}]
    seq = score_round(items, _FakeTok(), judge)
    judged.clear()
    par = score_round(items, _FakeTok(), judge, max_workers=4)
    assert [r.seed for r in par] == [r.seed for r in seq] == [1, 2]
    assert sorted(judged) == ["boom/lm_calls.jsonl", "hi/lm_calls.jsonl", "lo/lm_calls.jsonl"]
    assert np.allclose(np.asarray(par[0].token_adv), np.asarray(seq[0].token_adv))
    assert np.asarray(par[0].token_adv)[1] > np.asarray(par[1].token_adv)[1]
