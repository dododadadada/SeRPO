"""Tests for grpo.preprocess.tokenize_trajectory."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from grpo.preprocess.tokenize_trajectory import tokenize_trajectory

FIXTURE = Path(__file__).parent / "fixtures" / "seed_1_07b42fd_1_lm_calls.jsonl"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


def test_tokenize_returns_expected_shape(tokenizer):
    result = tokenize_trajectory(FIXTURE, tokenizer)
    assert "input_ids" in result
    assert "response_mask" in result
    assert "step_token_ranges" in result
    assert "num_steps" in result
    assert len(result["input_ids"]) == len(result["response_mask"])
    assert all(m in (0, 1) for m in result["response_mask"])


def test_prefix_masked_zero(tokenizer):
    """Tokens before the first assistant turn should have response_mask=0."""
    result = tokenize_trajectory(FIXTURE, tokenizer)
    # First step starts somewhere in the middle; leading tokens are prefix → 0
    assert result["response_mask"][0] == 0
    assert sum(result["response_mask"]) > 0


def test_step_ranges_cover_assistant_tokens_only(tokenizer):
    """Each step_token_range should map entirely to response_mask=1 spans."""
    result = tokenize_trajectory(FIXTURE, tokenizer)
    mask = result["response_mask"]
    for start, end in result["step_token_ranges"]:
        assert end > start, f"empty step range: ({start}, {end})"
        assert end <= len(mask)
        for i in range(start, end):
            assert mask[i] == 1, (
                f"position {i} in step range [{start}:{end}] has response_mask=0"
            )


def test_step_count_matches_lm_calls(tokenizer):
    """Number of steps should equal the number of LM calls."""
    result = tokenize_trajectory(FIXTURE, tokenizer)
    with FIXTURE.open() as f:
        n_calls = sum(1 for line in f if line.strip())
    assert result["num_steps"] == n_calls
    assert len(result["step_token_ranges"]) == n_calls


def test_decoded_step1_contains_known_token(tokenizer):
    """Decoding tokens of step 1 should yield text matching the original assistant content."""
    result = tokenize_trajectory(FIXTURE, tokenizer)
    s0, e0 = result["step_token_ranges"][0]
    decoded = tokenizer.decode(result["input_ids"][s0:e0])
    # Step 1 of 07b42fd_1: agent logs in to Spotify via supervisor passwords
    decoded_lower = decoded.lower()
    assert "spotify" in decoded_lower or "supervisor" in decoded_lower


def test_missing_marker_raises(tokenizer, tmp_path):
    """If the prefix marker is missing, raise ValueError."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "input": {
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "hi"},
                    ]
                }
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="prefix marker"):
        tokenize_trajectory(bad, tokenizer)
