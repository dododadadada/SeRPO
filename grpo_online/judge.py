"""Rubric (KS_baseline) judge over an external chat API.

Returns per-segment contribution records for one trajectory. Prompt
construction and JSON schema mirror rubric_reward/run_rollouts_KS_baseline.py
(shared helpers live in rubric_reward/prompt_utils.py).

Segment schema consumed downstream by compute_serpo_advantage:
    [{"start_step": int, "end_step": int, "contribution": float}, ...]
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable

logger = logging.getLogger("online")


# ---------------------------------------------------------------------------
# Parser (unit-tested; pure function, no I/O)
# ---------------------------------------------------------------------------

def parse_judge_response(raw: str) -> list[dict]:
    """Parse the LLM's response into a list of segment dicts.

    The KS_baseline prompt instructs the model to respond with a JSON array
    (not a wrapped object), e.g.::

        [
          {"start_step": 1, "end_step": 2, "subgoal": "...", "type": "...",
           "contribution": 4, "rationale": "..."},
          ...
        ]

    This function strips optional markdown fences, parses the JSON, and
    returns only the three fields that compute_serpo_advantage needs.
    """
    text = raw.strip()

    # Strip markdown fences (some models wrap JSON in ```json ... ```)
    if text.startswith("```"):
        # e.g. ```json\n[...]\n```  or  ```\n[...]\n```
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0].strip()

    segments = json.loads(text)
    out: list[dict] = []
    for s in segments:
        out.append({
            "start_step": int(s["start_step"]),
            "end_step":   int(s["end_step"]),
            "contribution": float(s["contribution"]),
        })
    return out


# ---------------------------------------------------------------------------
# Factory (integration; not called in unit tests)
# ---------------------------------------------------------------------------

def make_api_judge(cfg) -> Callable[[str], list[dict]]:
    """Return a judge_fn(lm_calls_path) -> list[{start_step, end_step, contribution}].

    Args:
        cfg: OnlineConfig (or any object with rubric_api_model, rubric_api_key_env,
             rubric_api_base attributes).

    The returned function:
    1. Builds the KS_baseline rubric prompt from the trajectory files found
       relative to lm_calls_path (environment_io.md, instruction.txt,
       evaluation/report.md) via rubric_reward.prompt_utils.build_ks_baseline_prompt.
    2. Calls an OpenAI-compatible chat API with temperature=0.
    3. Parses the response with parse_judge_response.
    """
    from openai import OpenAI
    from rubric_reward.prompt_utils import build_ks_baseline_prompt

    api_key = os.environ[cfg.rubric_api_key_env]
    base_url = cfg.rubric_api_base or None  # empty string -> None
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = cfg.rubric_api_model

    def judge_fn(lm_calls_path: str) -> list[dict]:
        messages = build_ks_baseline_prompt(lm_calls_path)
        resp = call_with_backoff(
            lambda: client.chat.completions.create(
                model=model, messages=messages, temperature=0.0),
            max_attempts=max_attempts, base_sleep=base_sleep)
        return parse_judge_response(resp.choices[0].message.content)

    max_attempts = int(getattr(cfg, "judge_max_attempts", 8))
    base_sleep = float(getattr(cfg, "judge_backoff_s", 15.0))
    return judge_fn


def call_with_backoff(fn: Callable[[], object], max_attempts: int = 8,
                      base_sleep: float = 15.0, _sleep=time.sleep) -> object:
    """Retry fn() on transient API errors (429 rate limit, 5xx, connection) with
    linear backoff: sleep base_sleep * attempt between tries. A round's 48 judge
    prompts (~10k tokens each) exceed a 200k tokens/min budget when sent
    concurrently, so 429s are expected and must be waited out, not dropped.
    Non-transient errors (400s other than 429) propagate immediately."""
    from openai import (APIConnectionError, APITimeoutError, InternalServerError,
                        RateLimitError)
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError) as e:
            if attempt == max_attempts:
                raise
            wait = base_sleep * attempt
            logger.warning("judge: %s (attempt %d/%d) — retrying in %.0fs",
                           type(e).__name__, attempt, max_attempts, wait)
            _sleep(wait)
