"""Tokenize an AppWorld trajectory and produce per-step assistant token ranges.

Input:  appworld/.../seed_N/tasks/<tid>/logs/lm_calls.jsonl
Output: {input_ids, attention_mask, response_mask, step_token_ranges, num_steps}

Mask rule:
  - System / few-shot / task-instruction tokens: response_mask = 0
  - Step k assistant message tokens (k = 1..N): response_mask = 1
  - Step k env-output user message tokens: response_mask = 0

Algorithm:
  1. Resolve message-boundary char positions deterministically from message
     content lengths. The Qwen2.5 chat template wraps every message as
     ``<|im_start|>{role}\\n{content}<|im_end|>\\n``. If no system message
     is present in the messages list, the template injects a default system
     message at the start; we account for this by extracting that default
     once per tokenizer.
  2. Render the chat as a string once, then tokenize that string once with
     ``return_offsets_mapping=True`` to obtain char→token offsets.
  3. Map each message's char-start to a token index → message token spans.

  This is O(N) in total tokens, O(1) in repeated chat-template invocations,
  and robust to ``<|im_start|>``/``<|im_end|>`` substrings embedded inside
  message content (which would break naive token-id scanning).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PREFIX_MARKER = "Using these APIs, now generate code to solve the actual task:"

# Qwen2.5 special tokens (verified via tokenizer.convert_tokens_to_ids).
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645

_IM_START_LEN = len("<|im_start|>")  # 12
_IM_END_LEN = len("<|im_end|>")  # 10


def _extract_default_system_content(tokenizer) -> str:
    """Return the system-message content that the tokenizer's chat template
    injects when the messages list does not start with a system message. Qwen2.5
    injects 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'.
    Other Qwen-family models may differ."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    m = re.match(
        r"<\|im_start\|>system\n(.*?)<\|im_end\|>", rendered, re.DOTALL
    )
    if m is None:
        raise RuntimeError(
            "could not extract default system content from chat template render"
        )
    return m.group(1)


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
    prefix_end_orig = _find_prefix_end(messages)

    interaction = messages[prefix_end_orig + 1 :]
    if not interaction or interaction[0].get("role") != "assistant":
        raise ValueError(
            f"expected first interaction message to be assistant (step 1), "
            f"got role={interaction[0].get('role') if interaction else None!r}"
        )

    # 1. Augment with default system message if the messages list does not
    #    start with one — this mirrors what the chat template does internally
    #    and lets us account for the injected tokens deterministically.
    if not messages or messages[0].get("role") != "system":
        default_sys = _extract_default_system_content(tokenizer)
        augmented = [{"role": "system", "content": default_sys}] + list(messages)
        prefix_end = prefix_end_orig + 1
    else:
        augmented = list(messages)
        prefix_end = prefix_end_orig

    # 2. Compute char boundaries deterministically.
    char_boundaries: list[int] = []
    char_pos = 0
    for m in augmented:
        char_boundaries.append(char_pos)
        # Each message: <|im_start|>{role}\n{content}<|im_end|>\n
        char_pos += (
            _IM_START_LEN + len(m["role"]) + 1
            + len(m["content"])
            + _IM_END_LEN + 1
        )

    # 3. Render the chat once and verify length matches our accounting.
    chat_str = tokenizer.apply_chat_template(
        augmented, tokenize=False, add_generation_prompt=False,
    )
    if len(chat_str) != char_pos:
        raise RuntimeError(
            f"char accounting mismatch: computed {char_pos}, "
            f"actual chat_str len {len(chat_str)}"
        )

    # 4. Tokenize once with offset_mapping so we can map char→token positions.
    enc = tokenizer(
        chat_str,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = list(enc["input_ids"])
    offsets = enc["offset_mapping"]

    # 5. Map each char boundary to the FIRST token whose char span starts at
    #    or after that boundary. Single linear sweep across offsets.
    msg_token_starts: list[int] = []
    j = 0
    for cb in char_boundaries:
        while j < len(offsets) and offsets[j][0] < cb:
            j += 1
        msg_token_starts.append(j)
    msg_token_starts.append(len(input_ids))  # sentinel for last message's end

    # 6. Build response_mask and step_token_ranges. Skip messages in prefix.
    response_mask = [0] * len(input_ids)
    step_ranges: list[tuple[int, int]] = []
    for i, msg in enumerate(augmented):
        if i <= prefix_end:
            continue
        start = msg_token_starts[i]
        end = msg_token_starts[i + 1]
        if msg.get("role") == "assistant":
            for k in range(start, end):
                response_mask[k] = 1
            step_ranges.append((start, end))

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
