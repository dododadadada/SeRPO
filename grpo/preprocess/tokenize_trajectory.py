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


# Qwen3.5-family chat templates inject an empty think block before the final
# assistant turn's content. Qwen2.5 has no such block.
_THINK_BLOCK_LEN = len("<think>\n\n</think>\n\n")


def _extract_default_system_content(tokenizer) -> str | None:
    """Return the system-message content that the tokenizer's chat template
    injects when the messages list does not start with a system message. Qwen2.5
    injects 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'.
    Returns None if the template injects no default system message (e.g. Qwen3.5)."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    m = re.match(
        r"<\|im_start\|>system\n(.*?)<\|im_end\|>", rendered, re.DOTALL
    )
    if m is None:
        return None
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
      step_anchor_obs: list[str]  (length = num_steps; per-step the content of
                                   the message immediately preceding that step's
                                   assistant message — the task-instruction
                                   prefix for step 1, the env-output for later)
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
        if default_sys is None:
            # Template injects no default system message (e.g. Qwen3.5) —
            # use the messages as-is, no augmentation.
            augmented = list(messages)
            prefix_end = prefix_end_orig
        else:
            augmented = [{"role": "system", "content": default_sys}] + list(messages)
            prefix_end = prefix_end_orig + 1
    else:
        augmented = list(messages)
        prefix_end = prefix_end_orig

    # 2. Render the chat once and tokenize with offset mapping. Rather than
    #    assuming a rigid per-message char layout (which varies across chat
    #    templates — e.g. Qwen3.5 strips trailing whitespace from content and
    #    injects an empty think block on the final assistant turn), we locate
    #    each message's content text directly in the rendered string.
    chat_str = tokenizer.apply_chat_template(
        augmented, tokenize=False, add_generation_prompt=False,
    )
    enc = tokenizer(
        chat_str,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = list(enc["input_ids"])
    offsets = enc["offset_mapping"]

    def _char_to_token(char_idx: int, *, start: bool) -> int:
        """First token index at/after char_idx. start=True maps a span start
        (token whose span starts >= char_idx); used for both ends."""
        lo, hi = 0, len(offsets)
        while lo < hi:
            mid = (lo + hi) // 2
            if offsets[mid][0] < char_idx:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # 3. Walk messages in order; locate each content in chat_str with a moving
    #    cursor. Mark assistant tokens of steps 1..N (i.e. after the prefix).
    response_mask = [0] * len(input_ids)
    step_ranges: list[tuple[int, int]] = []
    step_anchor_obs: list[str] = []
    cursor = 0
    prev_content = ""  # content of the message immediately before current
    for i, msg in enumerate(augmented):
        content = msg.get("content", "")
        target = content.rstrip()
        if not target:
            continue
        pos = chat_str.find(target, cursor)
        if pos < 0:
            pos = chat_str.find(content, cursor)
            target = content
        if pos < 0:
            raise RuntimeError(
                f"message {i} ({msg.get('role')}) content not found in render"
            )
        cstart, cend = pos, pos + len(target)
        cursor = cend
        if i > prefix_end and msg.get("role") == "assistant":
            tstart = _char_to_token(cstart, start=True)
            tend = _char_to_token(cend, start=True)
            for k in range(tstart, tend):
                response_mask[k] = 1
            step_ranges.append((tstart, tend))
            step_anchor_obs.append(prev_content)
        prev_content = content

    if len(input_ids) != len(response_mask):
        raise RuntimeError(
            f"length mismatch: input_ids={len(input_ids)} mask={len(response_mask)}"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "response_mask": response_mask,
        "step_token_ranges": step_ranges,
        "step_anchor_obs": step_anchor_obs,
        "num_steps": len(step_ranges),
    }
