"""Tokenize an AppWorld trajectory and produce per-step assistant token ranges.

Input:  appworld/.../seed_N/tasks/<tid>/logs/lm_calls.jsonl
Output: {input_ids, attention_mask, response_mask, step_token_ranges, num_steps}

Mask rule:
  - System / few-shot / task-instruction tokens: response_mask = 0
  - Step k assistant message tokens (k = 1..N): response_mask = 1
  - Step k env-output user message tokens: response_mask = 0
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREFIX_MARKER = "Using these APIs, now generate code to solve the actual task:"


def _read_last_messages(lm_calls_path: Path) -> list[dict[str, str]]:
    """Read the full message sequence from the LAST line: cumulative
    ``input.messages`` plus the response in ``output.choices[0].message``.

    The cumulative-input convention in AppWorld lm_calls.jsonl means the LAST
    line's ``input`` contains all turns BEFORE the final assistant response;
    that final response lives in ``output``. We append it here so the returned
    list reflects the complete trajectory (system + few-shot + task + N agent
    turns + N or N-1 env-output turns).
    """
    last: str | None = None
    with lm_calls_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        raise ValueError(f"empty lm_calls file: {lm_calls_path}")
    rec = json.loads(last)
    msgs = list(rec.get("input", {}).get("messages") or [])
    if not msgs:
        raise ValueError(f"no input.messages in last line of {lm_calls_path}")
    output = rec.get("output")
    if isinstance(output, dict):
        choices = output.get("choices") or []
        if choices:
            out_msg = choices[0].get("message")
            if isinstance(out_msg, dict) and out_msg.get("content"):
                msgs.append(
                    {
                        "role": out_msg.get("role", "assistant"),
                        "content": out_msg["content"],
                    }
                )
    return msgs


def _find_prefix_end(messages: list[dict[str, str]]) -> int:
    """Return the index of the last prefix message — the user message whose
    content contains PREFIX_MARKER (the actual task instruction follows after
    that marker, still inside the same user message)."""
    for i in reversed(range(len(messages))):
        m = messages[i]
        if m.get("role") == "user" and PREFIX_MARKER in m.get("content", ""):
            return i
    raise ValueError(
        f"prefix marker not found in messages "
        f"(expected substring: {PREFIX_MARKER!r})"
    )


def tokenize_trajectory(lm_calls_path: Path, tokenizer) -> dict[str, Any]:
    """Tokenize one trajectory and produce per-step assistant token ranges.

    Returns a dict with keys:
      input_ids:      list[int]
      attention_mask: list[int]   (all 1)
      response_mask:  list[int]   (1 on assistant tokens of steps 1..N only)
      step_token_ranges: list[(start, end)]  (length = num_steps)
      num_steps:      int
    """
    lm_calls_path = Path(lm_calls_path)
    messages = _read_last_messages(lm_calls_path)
    prefix_end = _find_prefix_end(messages)
    interaction = messages[prefix_end + 1 :]

    if not interaction or interaction[0].get("role") != "assistant":
        raise ValueError(
            f"expected first interaction message to be assistant (step 1), "
            f"got role={interaction[0].get('role') if interaction else None!r}"
        )

    # Tokenize incrementally so we can record exact token ranges per step.
    # We rely on the chat template being append-only across messages — i.e.
    # apply_chat_template(messages[:k+1]) starts with the same tokens as
    # apply_chat_template(messages[:k]). This holds for Qwen2.5's template
    # (each role tag opens/closes deterministically per message).
    prefix_tokens = tokenizer.apply_chat_template(
        messages[: prefix_end + 1],
        tokenize=True,
        add_generation_prompt=False,
    )
    input_ids: list[int] = list(prefix_tokens)
    response_mask: list[int] = [0] * len(prefix_tokens)
    step_ranges: list[tuple[int, int]] = []

    cumulative = list(messages[: prefix_end + 1])
    for msg in interaction:
        cumulative.append(msg)
        new_full = tokenizer.apply_chat_template(
            cumulative,
            tokenize=True,
            add_generation_prompt=False,
        )
        added_start = len(input_ids)
        added_end = len(new_full)
        if new_full[:added_start] != input_ids:
            mismatch = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(new_full, input_ids))
                    if a != b
                ),
                min(len(new_full), len(input_ids)),
            )
            raise RuntimeError(
                "chat template tokenization is not append-only; cannot align "
                f"step ranges. Diff at index {mismatch}."
            )
        input_ids = list(new_full)
        is_asst = msg.get("role") == "assistant"
        response_mask.extend(
            [1 if is_asst else 0] * (added_end - added_start)
        )
        if is_asst:
            step_ranges.append((added_start, added_end))

    if len(input_ids) != len(response_mask):
        raise RuntimeError(
            f"length mismatch: input_ids={len(input_ids)} mask={len(response_mask)}"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "response_mask": response_mask,
        "step_token_ranges": step_ranges,
        "num_steps": len(step_ranges),
    }
