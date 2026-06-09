"""Unit tests for grpo_online.judge (pure parser only; no live API calls)."""
import json
import pytest
from grpo_online.judge import parse_judge_response


def test_parse_judge_response_basic():
    """LLM returns a JSON array of segments (NOT a wrapped object)."""
    raw = json.dumps([
        {"start_step": 1, "end_step": 2, "subgoal": "explore spotify api",
         "type": "api_exploration", "contribution": 4,
         "rationale": "Efficiently explored available APIs."},
        {"start_step": 3, "end_step": 3, "subgoal": "login to spotify",
         "type": "login", "contribution": 1,
         "rationale": "Login failed due to wrong credentials."},
    ])
    segs = parse_judge_response(raw)
    assert segs == [
        {"start_step": 1, "end_step": 2, "contribution": 4.0},
        {"start_step": 3, "end_step": 3, "contribution": 1.0},
    ]


def test_parse_judge_response_strips_extra_fields():
    """Extra fields (subgoal, type, rationale) are dropped; only the three
    fields that compute_serpo_advantage consumes are kept."""
    raw = json.dumps([
        {"start_step": 2, "end_step": 5, "subgoal": "fetch data",
         "type": "data_fetch", "contribution": 5, "rationale": "Fast fetch."},
    ])
    segs = parse_judge_response(raw)
    assert len(segs) == 1
    assert set(segs[0].keys()) == {"start_step", "end_step", "contribution"}


def test_parse_judge_response_contribution_is_float():
    """contribution must be float (compute_serpo_advantage casts via float())."""
    raw = json.dumps([{"start_step": 1, "end_step": 1, "contribution": 3,
                       "type": "action", "subgoal": "do thing", "rationale": "ok"}])
    segs = parse_judge_response(raw)
    assert isinstance(segs[0]["contribution"], float)


def test_parse_judge_response_markdown_fence_stripped():
    """Some models wrap JSON in markdown fences; parser must handle it."""
    inner = json.dumps([
        {"start_step": 1, "end_step": 2, "contribution": 2,
         "type": "error_recovery", "subgoal": "recover", "rationale": "meh"},
    ])
    raw = f"```json\n{inner}\n```"
    segs = parse_judge_response(raw)
    assert segs[0]["start_step"] == 1
    assert segs[0]["contribution"] == 2.0


def test_parse_judge_response_empty_array():
    """Empty array should return empty list (boundary condition)."""
    segs = parse_judge_response("[]")
    assert segs == []
