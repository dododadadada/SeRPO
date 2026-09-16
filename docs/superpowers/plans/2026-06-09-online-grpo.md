# Online (semi-online) GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an on-policy (option-A, M=6×G=8×K=1/round) online GRPO loop for Qwen3.5-9B + LoRA on AppWorld that re-syncs the peft generation server to the trainer every round, reusing the offline trainer, advantage code, rubric reward, and AppWorld rollout.

**Architecture:** Two processes on 3 GPUs — a trainer (GPU 0,1, holds the 9B+LoRA+optimizer and takes grad steps via the offline `grpo_loss_step`) and a peft generation server (GPU 2, serves base+current LoRA over an OpenAI-compatible endpoint with a `/reload_adapter` hot-swap). An orchestrator runs the round loop: rollout (AppWorld via the peft server) → reward (rubric API judge) → serpo advantages → K grad steps → write LoRA → reload generator. Judge is an external API (no GPU).

**Tech Stack:** Python, PyTorch, transformers, peft, FastAPI/uvicorn (gen server), pyyaml; reuses `grpo/trainer/offline_trainer.py`, `grpo/preprocess/make_advantages.py`, `grpo/preprocess/tokenize_trajectory.py`, `grpo/eval/peft_chat_server.py`, `rubric_reward/`, AppWorld `appworld run`.

---

## File Structure

| File | Responsibility |
|---|---|
| `grpo_online/__init__.py` | package marker |
| `grpo_online/config.py` | `OnlineConfig` dataclass + YAML loader |
| `grpo_online/gen_server.py` | peft server + `/reload_adapter` (extends `peft_chat_server`) |
| `grpo_online/reward.py` | new trajectories → rubric API contributions → `RolloutData` with serpo advantages |
| `grpo_online/online_trainer.py` | `OnlineTrainer`: holds model/optimizer; `train_on_batch(rollouts, K)` |
| `grpo_online/online_loop.py` | orchestrator: round loop + task sampling + budget/stop |
| `grpo_online/config/online_v1.yaml` | concrete config |
| `grpo_online/tests/test_*.py` | unit tests (pure logic + mocked API/model) |

Reused unchanged: `grpo_loss_step`, `compute_serpo_advantage`, `tokenize_trajectory`, `collate_pad_right`.

---

## Task 1: Package scaffold + config

**Files:**
- Create: `grpo_online/__init__.py`, `grpo_online/config.py`
- Test: `grpo_online/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# grpo_online/tests/test_config.py
from grpo_online.config import OnlineConfig, load_config

def test_defaults():
    c = OnlineConfig()
    assert c.M == 6 and c.G == 8 and c.K == 1
    assert c.lr == 1e-6 and c.kl_beta == 0.04
    assert c.gen_ports == [8101]

def test_yaml_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("M: 4\nK: 2\ngen_ports: [8101, 8102]\n")
    c = load_config(str(p))
    assert c.M == 4 and c.K == 2 and c.gen_ports == [8101, 8102]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grpo_online/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: grpo_online.config`

- [ ] **Step 3: Write minimal implementation**

```python
# grpo_online/__init__.py
```
```python
# grpo_online/config.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import yaml

@dataclass
class OnlineConfig:
    model_name: str = "Qwen/Qwen3.5-9B"
    # round shape
    M: int = 6              # tasks per round
    G: int = 8              # samples per task
    K: int = 1              # grad steps per round
    N_rounds: int = 76      # total rounds (budget)
    dataset: str = "train"
    temperature: float = 1.0
    top_p: float = 1.0
    # trainer HPs (mirror v3i)
    lr: float = 1e-6
    kl_beta: float = 0.04
    clip_eps: float = 0.2
    grad_clip: float = 1.0
    micro_batch_size: int = 1
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    kl_per_token_cap: float = 10.0
    outlier_logratio_threshold: float = 3.0
    max_masked_fraction: float = 0.05
    kl_halt_threshold: float = 5.0
    ratio_halt_threshold: float = 30.0
    seed: int = 42
    gradient_checkpointing: bool = True
    # infra
    trainer_gpus: str = "0,1"
    gen_gpus: list = field(default_factory=lambda: ["2"])
    gen_ports: list = field(default_factory=lambda: [8101])
    adapter_dir: str = "grpo_online/runs/v1/adapter_current"
    output_dir: str = "grpo_online/runs/v1"
    # reward (rubric API judge)
    rubric_api_model: str = "gpt-4o-mini"
    rubric_api_base: str = ""          # empty = default openai
    rubric_api_key_env: str = "OPENAI_API_KEY"
    rubric_variant: str = "KS_baseline"

def load_config(path: str) -> OnlineConfig:
    raw = yaml.safe_load(open(path).read()) or {}
    base = asdict(OnlineConfig())
    base.update(raw)
    if isinstance(base.get("lora_target_modules"), list):
        base["lora_target_modules"] = tuple(base["lora_target_modules"])
    return OnlineConfig(**base)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest grpo_online/tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add grpo_online/__init__.py grpo_online/config.py grpo_online/tests/test_config.py
git commit -m "feat(online): package scaffold + OnlineConfig"
```

---

## Task 2: Generation server with `/reload_adapter`

**Files:**
- Create: `grpo_online/gen_server.py`
- Test: `grpo_online/tests/test_gen_server.py` (uses a tiny HF model, CPU — no Qwen download)

`gen_server.py` reuses `peft_chat_server`'s request handling but adds (a) a `/reload_adapter` endpoint that hot-swaps the LoRA in place and (b) startup that loads base + an initial adapter dir.

- [ ] **Step 1: Write the failing test**

```python
# grpo_online/tests/test_gen_server.py
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from grpo_online.gen_server import reload_adapter_into

def _tiny_peft(tmp_path):
    m = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    pm = get_peft_model(m, LoraConfig(r=4, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM"))
    return pm

def test_reload_adapter_swaps_weights(tmp_path):
    pm = _tiny_peft(tmp_path)
    # save adapter A
    a = tmp_path / "A"; pm.save_pretrained(str(a))
    # mutate the live LoRA-B and save as adapter B (different weights)
    for n, p in pm.named_parameters():
        if "lora_B" in n: p.data.add_(1.0)
    b = tmp_path / "B"; pm.save_pretrained(str(b))
    # load A back into the live model, then reload B via the function under test
    pm.load_adapter(str(a), adapter_name="default", is_trainable=False)
    pm.set_adapter("default")
    before = [p.detach().clone() for n, p in pm.named_parameters() if "lora_B" in n][0]
    reload_adapter_into(pm, str(b))
    after = [p for n, p in pm.named_parameters() if "lora_B" in n][0]
    assert not torch.allclose(before, after)  # weights changed to B
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grpo_online/tests/test_gen_server.py -v`
Expected: FAIL with `ModuleNotFoundError: grpo_online.gen_server`

- [ ] **Step 3: Write minimal implementation**

```python
# grpo_online/gen_server.py
"""peft generation server with hot-reloadable LoRA adapter (online GRPO).

Serves base + current LoRA over an OpenAI-compatible /v1/chat/completions, plus
POST /reload_adapter {"path": "..."} to hot-swap the LoRA in place (~seconds) so
the orchestrator can re-sync after each training round. Reuses the inference
handler shape of grpo/eval/peft_chat_server.py; defaults enable_thinking=False.

Usage:
  CUDA_VISIBLE_DEVICES=2 python -m grpo_online.gen_server \
    --base-model Qwen/Qwen3.5-9B --adapter <dir> --served-name Qwen/Qwen3.5-9B --port 8101
"""
from __future__ import annotations
import argparse, asyncio, time, uuid
from typing import Any
import torch, uvicorn
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL = None; _TOK = None; _NAME = ""; _LOCK = asyncio.Lock()


def reload_adapter_into(peft_model, adapter_dir: str) -> None:
    """Hot-swap the LoRA weights of `peft_model` to those saved at adapter_dir.
    Reloads the 'default' adapter in place (same module targets/rank assumed)."""
    peft_model.load_adapter(adapter_dir, adapter_name="default", is_trainable=False)
    peft_model.set_adapter("default")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[dict]
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    chat_template_kwargs: dict | None = None


def _infer(req: ChatRequest) -> dict[str, Any]:
    ckw = {"enable_thinking": False}
    if req.chat_template_kwargs:
        ckw.update(req.chat_template_kwargs)
    try:
        prompt = _TOK.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True, **ckw)
    except TypeError:
        prompt = _TOK.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
    inputs = _TOK(prompt, return_tensors="pt").to(_MODEL.device)
    plen = int(inputs.input_ids.shape[1])
    maxn = int(req.max_completion_tokens or req.max_tokens or 3000)
    temp = float(req.temperature or 0.0); do_sample = temp > 0.0
    gk = {"max_new_tokens": maxn, "do_sample": do_sample, "pad_token_id": _TOK.eos_token_id}
    if do_sample:
        gk["temperature"] = temp; gk["top_p"] = float(req.top_p) if req.top_p is not None else 1.0
    with torch.inference_mode():
        out = _MODEL.generate(**inputs, **gk)
    new = out[0, plen:]; text = _TOK.decode(new, skip_special_tokens=True)
    n = int(new.size(0))
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion",
            "created": int(time.time()), "model": req.model or _NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "length" if n >= maxn else "stop"}],
            "usage": {"prompt_tokens": plen, "completion_tokens": n, "total_tokens": plen + n}}


app = FastAPI()

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/reload_adapter")
async def reload_adapter(body: dict):
    async with _LOCK:
        reload_adapter_into(_MODEL, body["path"])
    return {"status": "reloaded", "path": body["path"]}

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    async with _LOCK:
        return _infer(req)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--device-map", default="cuda:0")
    args = ap.parse_args()
    global _MODEL, _TOK, _NAME
    _NAME = args.served_name
    _TOK = AutoTokenizer.from_pretrained(args.base_model)
    if _TOK.pad_token_id is None: _TOK.pad_token_id = _TOK.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map=args.device_map)
    base.eval()
    _MODEL = PeftModel.from_pretrained(base, args.adapter, adapter_name="default")
    _MODEL.eval()
    print(f"ready: serving '{_NAME}' on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest grpo_online/tests/test_gen_server.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add grpo_online/gen_server.py grpo_online/tests/test_gen_server.py
git commit -m "feat(online): peft gen server with /reload_adapter hot-swap"
```

---

## Task 3: Reward — trajectories → contributions → serpo advantages

**Files:**
- Create: `grpo_online/reward.py`
- Test: `grpo_online/tests/test_reward.py` (mocks the rubric API; uses a real tiny tokenizer)

`reward.py` exposes `score_round(task_seed_paths, tokenizer, judge_fn) -> list[RolloutData]`:
for each (task_id, seed) it tokenizes the trajectory, calls `judge_fn(lm_calls_path) -> segments` (the rubric API judge; injected so it's mockable), attaches segments, then groups by task and calls `compute_serpo_advantage` per group. Returns flat RolloutData with `.token_adv` set.

- [ ] **Step 1: Write the failing test**

```python
# grpo_online/tests/test_reward.py
import numpy as np
from grpo_online.reward import score_round

class _FakeTok:
    pad_token_id = 0

def test_score_round_sets_segment_advantages(monkeypatch):
    # two rollouts of one task; fake tokenize + fake judge
    def fake_tok(path, tokenizer):
        return {"input_ids": [1,2,3,4], "attention_mask": [1,1,1,1],
                "response_mask": [0,1,1,0], "step_token_ranges": [(1,3)]}
    def fake_judge(path):
        return [{"contribution": 5 if "a" in str(path) else 1,
                 "start_step": 1, "end_step": 1}]
    monkeypatch.setattr("grpo_online.reward.tokenize_trajectory", fake_tok)
    items = [{"task_id": "t1", "seed": 1, "lm_calls_path": "a/lm_calls.jsonl", "outcome": 1.0},
             {"task_id": "t1", "seed": 2, "lm_calls_path": "b/lm_calls.jsonl", "outcome": 0.0}]
    rds = score_round(items, _FakeTok(), fake_judge)
    assert len(rds) == 2
    a0 = np.asarray(rds[0].token_adv); a1 = np.asarray(rds[1].token_adv)
    # the high-contribution rollout gets a higher (positive) z-scored advantage
    assert a0[1] > a1[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grpo_online/tests/test_reward.py -v`
Expected: FAIL with `ModuleNotFoundError: grpo_online.reward`

- [ ] **Step 3: Write minimal implementation**

```python
# grpo_online/reward.py
"""Online reward: tokenize new trajectories, score segments via a rubric judge,
and compute serpo per-segment advantages per task group, in memory."""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Callable
from grpo.preprocess.make_advantages import RolloutData, compute_serpo_advantage
from grpo.preprocess.tokenize_trajectory import tokenize_trajectory  # patched in tests

MAX_TOKENS = 32000


def score_round(items: list[dict[str, Any]], tokenizer,
                judge_fn: Callable[[str], list[dict]]) -> list[RolloutData]:
    """items: dicts with task_id, seed, lm_calls_path, outcome.
    judge_fn(lm_calls_path) -> list of {contribution, start_step, end_step}."""
    rds: list[RolloutData] = []
    by_task: dict[str, list[RolloutData]] = defaultdict(list)
    for it in items:
        try:
            tok = tokenize_trajectory(it["lm_calls_path"], tokenizer)
        except Exception:
            continue
        if len(tok["input_ids"]) > MAX_TOKENS:
            continue
        segments = judge_fn(it["lm_calls_path"])
        rd = RolloutData(
            task_id=it["task_id"], seed=it["seed"],
            input_ids=tok["input_ids"], attention_mask=tok["attention_mask"],
            response_mask=tok["response_mask"], step_token_ranges=tok["step_token_ranges"],
            segments=segments, outcome=float(it.get("outcome", 0.0)),
        )
        by_task[it["task_id"]].append(rd)
    for group in by_task.values():
        compute_serpo_advantage(group)  # mutates .token_adv per rollout
        rds.extend(group)
    return rds
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest grpo_online/tests/test_reward.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add grpo_online/reward.py grpo_online/tests/test_reward.py
git commit -m "feat(online): in-memory serpo reward from trajectories + judge_fn"
```

---

## Task 4: Rubric API judge adapter

**Files:**
- Create: `grpo_online/judge.py`
- Test: `grpo_online/tests/test_judge.py` (mocks the OpenAI client; verifies prompt assembly + parse)

`judge.py` exposes `make_api_judge(cfg) -> judge_fn`. The returned `judge_fn(lm_calls_path)` reads the trajectory, builds the KS_baseline rubric prompt (reuse `rubric_reward`'s prompt builder + segment parser), calls the API model, and returns `[{contribution, start_step, end_step}, ...]`. The API client is injected/created from `cfg.rubric_api_*`.

- [ ] **Step 1: Write the failing test**

```python
# grpo_online/tests/test_judge.py
from grpo_online.judge import parse_judge_response

def test_parse_judge_response():
    # the judge returns JSON segments; parser normalizes to our schema
    raw = '{"segments": [{"start_step": 1, "end_step": 2, "contribution": 4}, {"start_step": 3, "end_step": 3, "contribution": 1}]}'
    segs = parse_judge_response(raw)
    assert segs == [{"start_step": 1, "end_step": 2, "contribution": 4},
                    {"start_step": 3, "end_step": 3, "contribution": 1}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grpo_online/tests/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: grpo_online.judge`

- [ ] **Step 3: Write minimal implementation**

> NOTE: inspect `rubric_reward/run_rollouts.py` for the exact prompt template and
> the segment JSON schema it expects, and import its prompt builder if one exists
> (e.g. `from rubric_reward.prompts import build_ks_baseline_prompt`). If the
> rubric code only supports a local-model client, factor its prompt-build + parse
> into importable functions and call the API here. The parser below assumes the
> judge emits `{"segments": [{start_step, end_step, contribution}, ...]}`.

```python
# grpo_online/judge.py
"""Rubric (KS_baseline) judge over an external chat API. Returns per-segment
contribution records for a trajectory. Prompt/parse mirror rubric_reward."""
from __future__ import annotations
import json, os
from typing import Callable

def parse_judge_response(raw: str) -> list[dict]:
    obj = json.loads(raw)
    out = []
    for s in obj["segments"]:
        out.append({"start_step": int(s["start_step"]),
                    "end_step": int(s["end_step"]),
                    "contribution": float(s["contribution"])})
    return out

def make_api_judge(cfg) -> Callable[[str], list[dict]]:
    from openai import OpenAI
    from rubric_reward.run_rollouts import build_ks_baseline_prompt  # reuse prompt
    client = OpenAI(api_key=os.environ[cfg.rubric_api_key_env],
                    base_url=cfg.rubric_api_base or None)
    def judge_fn(lm_calls_path: str) -> list[dict]:
        messages = build_ks_baseline_prompt(lm_calls_path)
        resp = client.chat.completions.create(
            model=cfg.rubric_api_model, messages=messages, temperature=0.0,
            response_format={"type": "json_object"})
        return parse_judge_response(resp.choices[0].message.content)
    return judge_fn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest grpo_online/tests/test_judge.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add grpo_online/judge.py grpo_online/tests/test_judge.py
git commit -m "feat(online): rubric API judge adapter (KS_baseline)"
```

---

## Task 5: OnlineTrainer — hold model, train_on_batch

**Files:**
- Create: `grpo_online/online_trainer.py`
- Test: `grpo_online/tests/test_online_trainer.py` (tiny GPT2 + tiny LoRA, CPU; asserts a step changes LoRA weights and saves)

`OnlineTrainer` builds the peft model + optimizer once (reusing the offline trainer's model setup helpers where possible). `train_on_batch(rollouts, K)` collates the RolloutData into the batch dict `grpo_loss_step` expects, computes old/ref logp fresh (adapter on / off), then runs K grad steps, returning metrics. `save_adapter(path)` writes the LoRA for the generator to reload.

- [ ] **Step 1: Write the failing test**

```python
# grpo_online/tests/test_online_trainer.py
import torch
from grpo_online.online_trainer import OnlineTrainer
from grpo.preprocess.make_advantages import RolloutData

def _batch():
    r = RolloutData(task_id="t", seed=1, input_ids=[1,2,3,4,5],
                    attention_mask=[1,1,1,1,1], response_mask=[0,1,1,1,0],
                    step_token_ranges=[(1,4)], segments=[], outcome=1.0)
    import numpy as np
    r.token_adv = np.array([0,0.5,0.5,0.5,0], dtype="float32")
    return [r]

def test_train_step_updates_and_saves(tmp_path):
    tr = OnlineTrainer(model_name="hf-internal-testing/tiny-random-gpt2",
                       lora_target_modules=["c_attn"], device_map="cpu")
    before = tr.lora_snapshot()
    m = tr.train_on_batch(_batch(), K=1)
    assert "loss" in m and "kl_loss" in m
    after = tr.lora_snapshot()
    assert any(not torch.allclose(before[k], after[k]) for k in before)
    tr.save_adapter(str(tmp_path / "ad"))
    assert (tmp_path / "ad" / "adapter_model.safetensors").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grpo_online/tests/test_online_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError: grpo_online.online_trainer`

- [ ] **Step 3: Write minimal implementation**

> Reuse from `grpo/trainer/offline_trainer.py`: `grpo_loss_step`, `shift_for_causal_lm`,
> `_forward_logprobs`, `GRPOConfig`. Build the batch dict with keys `input_ids,
> attention_mask, response_mask, advantages, old_logp, ref_logp` (right-padded via
> `grpo.trainer.dataset.collate_pad_right`). Compute old_logp (adapter on) and
> ref_logp (adapter off, `model.disable_adapter()`) under `torch.no_grad()`.

```python
# grpo_online/online_trainer.py
from __future__ import annotations
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from grpo.trainer.offline_trainer import (
    GRPOConfig, grpo_loss_step, shift_for_causal_lm, _forward_logprobs,
)
from grpo.trainer.dataset import collate_pad_right

class OnlineTrainer:
    def __init__(self, model_name, lora_target_modules, lr=1e-6, lora_rank=32,
                 lora_alpha=32, device_map="auto", gradient_checkpointing=True, **gcfg):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map=device_map)
        if gradient_checkpointing:
            base.gradient_checkpointing_enable(); base.enable_input_require_grads()
        base.config.use_cache = False
        self.model = get_peft_model(base, LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, target_modules=list(lora_target_modules),
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
        self.model.train()
        self.cfg = GRPOConfig(model_name=model_name, lr=lr, lora_rank=lora_rank,
                              lora_alpha=lora_alpha,
                              lora_target_modules=tuple(lora_target_modules), **gcfg)
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr)

    def _collate(self, rollouts):
        batch = collate_pad_right([{
            "input_ids": r.input_ids, "attention_mask": r.attention_mask,
            "response_mask": r.response_mask, "advantages": r.token_adv.tolist(),
            "task_ids": r.task_id, "seeds": r.seed} for r in rollouts], self.tok.pad_token_id)
        inputs, labels, attn, _, _ = shift_for_causal_lm(
            batch["input_ids"], batch["attention_mask"], batch["response_mask"], batch["advantages"])
        with torch.no_grad():
            old = _forward_logprobs(self.model, inputs, attn, labels).cpu()
            with self.model.disable_adapter():
                ref = _forward_logprobs(self.model, inputs, attn, labels).cpu()
        batch["old_logp"] = old; batch["ref_logp"] = ref
        return batch

    def train_on_batch(self, rollouts, K=1):
        batch = self._collate(rollouts)
        metrics = {}
        for _ in range(K):
            self.opt.zero_grad()
            metrics = grpo_loss_step(self.model, batch, self.cfg)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.cfg.grad_clip)
            self.opt.step()
        return metrics

    def lora_snapshot(self):
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if "lora_" in n}

    def save_adapter(self, path):
        self.model.save_pretrained(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest grpo_online/tests/test_online_trainer.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add grpo_online/online_trainer.py grpo_online/tests/test_online_trainer.py
git commit -m "feat(online): OnlineTrainer.train_on_batch over grpo_loss_step"
```

---

## Task 6: Rollout adapter (AppWorld via the gen server)

**Files:**
- Create: `grpo_online/rollout.py`
- Test: `grpo_online/tests/test_rollout.py` (unit-test only the pure helpers: task sampling + output-dir parsing; the actual `appworld run` is integration, covered in Task 8)

`rollout.py` exposes:
- `sample_tasks(all_tasks, M, rng) -> list[str]`
- `run_rollout(cfg, round_idx, tasks) -> list[dict]` — invokes `appworld run` (G seeds × tasks) against the gen server port(s), then returns items `{task_id, seed, lm_calls_path, outcome}` by reading the output dir (mirrors `grpo/preprocess/build_dataset` input assembly + `_load_outcomes`).

- [ ] **Step 1: Write the failing test**

```python
# grpo_online/tests/test_rollout.py
import random
from grpo_online.rollout import sample_tasks, items_from_outputs

def test_sample_tasks_no_repeat_within_round():
    ts = [f"t{i}" for i in range(20)]
    got = sample_tasks(ts, 6, random.Random(0))
    assert len(got) == 6 and len(set(got)) == 6 and set(got) <= set(ts)

def test_items_from_outputs(tmp_path):
    # build a fake output tree: seed_1/tasks/t1/logs/lm_calls.jsonl + evaluations/train.json
    seed = tmp_path / "seed_1"
    (seed / "tasks" / "t1" / "logs").mkdir(parents=True)
    (seed / "tasks" / "t1" / "logs" / "lm_calls.jsonl").write_text("{}\n")
    (seed / "evaluations").mkdir()
    (seed / "evaluations" / "train.json").write_text(
        '{"individual": {"t1": {"success": true, "passes": [1,2], "num_tests": 2}}}')
    items = items_from_outputs(str(tmp_path), seeds=[1], tasks=["t1"], outcome_type="continuous")
    assert items == [{"task_id": "t1", "seed": 1,
                      "lm_calls_path": str(seed / "tasks" / "t1" / "logs" / "lm_calls.jsonl"),
                      "outcome": 1.0}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grpo_online/tests/test_rollout.py -v`
Expected: FAIL with `ModuleNotFoundError: grpo_online.rollout`

- [ ] **Step 3: Write minimal implementation**

> `run_rollout` shells out to `appworld run` per seed against the gen port, writing
> to `cfg.output_dir/rollouts/round_{i}/seed_{s}` — mirror the env vars and flags in
> `run_round0_rollout_9b.sh` (ROLLOUT_SEED, ROLLOUT_TEMPERATURE, ROLLOUT_DATASET,
> VLLM_PORT, NO_API_KEY, `--num-processes`, `--with-evaluation --without-setup`).
> Restrict to the sampled `tasks` by writing a temp dataset file and pointing
> ROLLOUT_DATASET at it (verify how the rollout config selects task subsets first).

```python
# grpo_online/rollout.py
from __future__ import annotations
import json, os, subprocess
from pathlib import Path

def sample_tasks(all_tasks, M, rng):
    return rng.sample(list(all_tasks), min(M, len(all_tasks)))

def items_from_outputs(base, seeds, tasks, outcome_type="continuous"):
    items = []
    for s in seeds:
        ev = Path(base) / f"seed_{s}" / "evaluations" / "train.json"
        ind = json.loads(ev.read_text()).get("individual", {}) if ev.exists() else {}
        for t in tasks:
            lm = Path(base) / f"seed_{s}" / "tasks" / t / "logs" / "lm_calls.jsonl"
            if not lm.exists():
                continue
            rec = ind.get(t, {})
            if outcome_type == "continuous":
                npass, ntest = len(rec.get("passes", [])), int(rec.get("num_tests", 0))
                outcome = (npass / ntest) if ntest > 0 else 0.0
            else:
                outcome = 1.0 if rec.get("success") else 0.0
            items.append({"task_id": t, "seed": s, "lm_calls_path": str(lm), "outcome": outcome})
    return items

def run_rollout(cfg, round_idx, tasks, appworld_bin, experiment):
    base = Path(cfg.output_dir) / "rollouts" / f"round_{round_idx}"
    base.mkdir(parents=True, exist_ok=True)
    # subset dataset file (one task id per line) for this round
    ds = base / "tasks.txt"; ds.write_text("\n".join(tasks) + "\n")
    port = cfg.gen_ports[0]
    for s in range(1, cfg.G + 1):
        env = {**os.environ, "OPENAI_API_KEY": "dummy", "NO_API_KEY": "dummy",
               "ROLLOUT_SEED": str(s), "ROLLOUT_TEMPERATURE": str(cfg.temperature),
               "ROLLOUT_TOP_P": str(cfg.top_p), "ROLLOUT_DATASET": str(ds),
               "VLLM_PORT": str(port)}
        subprocess.run([appworld_bin, "run", experiment, "--num-processes", "4",
                        "--with-evaluation", "--without-setup"], env=env, check=False,
                       cwd=str(Path.cwd() / "appworld"))
        # appworld writes to experiments/outputs/<experiment>; move to seed dir
        out = Path("appworld/experiments/outputs") / experiment
        dst = base / f"seed_{s}"
        if dst.exists():
            import shutil; shutil.rmtree(dst)
        out.rename(dst)
    return items_from_outputs(str(base), list(range(1, cfg.G + 1)), tasks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest grpo_online/tests/test_rollout.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add grpo_online/rollout.py grpo_online/tests/test_rollout.py
git commit -m "feat(online): rollout adapter (sample tasks + appworld run + parse)"
```

---

## Task 7: Orchestrator loop

**Files:**
- Create: `grpo_online/online_loop.py`
- Test: `grpo_online/tests/test_online_loop.py` (inject fakes for rollout/judge/trainer/reload; assert ordering + sync + halt)

`run_online(cfg, deps)` runs N rounds: `tasks = sample_tasks` → `items = deps.rollout(round, tasks)` → `rds = score_round(items, tok, deps.judge)` → `metrics = deps.trainer.train_on_batch(rds, cfg.K)` → `deps.trainer.save_adapter(cfg.adapter_dir)` → `deps.reload_servers(cfg.adapter_dir)`. Stops on `metrics['kl_loss'] > cfg.kl_halt_threshold`. `deps` is a small struct so the loop is unit-testable with fakes.

- [ ] **Step 1: Write the failing test**

```python
# grpo_online/tests/test_online_loop.py
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
        score_round=lambda items, tok, judge: items,  # passthrough fake
    )
    cfg = type("C", (), {"M": 2, "K": 1, "N_rounds": 2, "adapter_dir": str(tmp_path/"ad"),
                         "kl_halt_threshold": 5.0, "seed": 0})()
    run_online(cfg, deps)
    # each round: rollout then reload, twice
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
    assert rounds == 1  # halted after round 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest grpo_online/tests/test_online_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: grpo_online.online_loop`

- [ ] **Step 3: Write minimal implementation**

```python
# grpo_online/online_loop.py
from __future__ import annotations
import logging, random
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("online")

@dataclass
class Deps:
    all_tasks: list
    rollout: Callable[[int, list], list]
    judge: Callable[[str], list]
    tokenizer: Any
    trainer: Any
    reload_servers: Callable[[str], None]
    score_round: Callable = None  # injected; defaults to reward.score_round

def run_online(cfg, deps: Deps) -> int:
    if deps.score_round is None:
        from grpo_online.reward import score_round as _sr
        deps.score_round = _sr
    rng = random.Random(cfg.seed)
    from grpo_online.rollout import sample_tasks
    done = 0
    for rnd in range(1, cfg.N_rounds + 1):
        tasks = sample_tasks(deps.all_tasks, cfg.M, rng)
        items = deps.rollout(rnd, tasks)
        rds = deps.score_round(items, deps.tokenizer, deps.judge)
        metrics = deps.trainer.train_on_batch(rds, cfg.K)
        done = rnd
        logger.info("round %d: loss=%.4f kl=%.4f (rds=%d)",
                    rnd, metrics.get("loss", 0.0), metrics.get("kl_loss", 0.0), len(rds))
        if metrics.get("kl_loss", 0.0) > cfg.kl_halt_threshold:
            logger.error("round %d kl=%.3f exceeded halt — stopping", rnd, metrics["kl_loss"])
            break
        deps.trainer.save_adapter(cfg.adapter_dir)
        deps.reload_servers(cfg.adapter_dir)
    return done
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest grpo_online/tests/test_online_loop.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add grpo_online/online_loop.py grpo_online/tests/test_online_loop.py
git commit -m "feat(online): orchestrator loop with sync + kl-halt (unit-tested with fakes)"
```

---

## Task 8: Entry point + config + GPU smoke test

**Files:**
- Create: `grpo_online/run_online.py`, `grpo_online/config/online_v1.yaml`, `grpo_online/scripts/launch.sh`
- Test: manual GPU smoke (documented), not pytest.

`run_online.py` wires real deps: launches/【expects】 gen server(s), builds `OnlineTrainer`, builds the API judge, builds `reload_servers` (POST `/reload_adapter` to each port), loads tasks from `appworld/data/datasets/train.txt`, calls `run_online`. `launch.sh` starts the gen server (GPU 2) then the loop (GPU 0,1).

- [ ] **Step 1: Write the entry point**

```python
# grpo_online/run_online.py
from __future__ import annotations
import argparse, requests
from grpo_online.config import load_config
from grpo_online.online_loop import run_online, Deps
from grpo_online.online_trainer import OnlineTrainer
from grpo_online.judge import make_api_judge
from grpo_online.rollout import run_rollout

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--appworld-bin", required=True)
    ap.add_argument("--experiment", default="rollout/online/qwen35_9b")
    args = ap.parse_args()
    cfg = load_config(args.config)
    trainer = OnlineTrainer(
        model_name=cfg.model_name, lora_target_modules=cfg.lora_target_modules,
        lr=cfg.lr, lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha, device_map="auto",
        kl_beta=cfg.kl_beta, clip_eps=cfg.clip_eps, grad_clip=cfg.grad_clip,
        micro_batch_size=cfg.micro_batch_size, kl_per_token_cap=cfg.kl_per_token_cap,
        outlier_logratio_threshold=cfg.outlier_logratio_threshold,
        max_masked_fraction=cfg.max_masked_fraction)
    # seed adapter so the gen server has something to load on boot
    trainer.save_adapter(cfg.adapter_dir)
    judge = make_api_judge(cfg)
    def reload_servers(path):
        for p in cfg.gen_ports:
            requests.post(f"http://localhost:{p}/reload_adapter", json={"path": path}, timeout=120)
    tasks = [l.strip() for l in open("appworld/data/datasets/train.txt") if l.strip()]
    deps = Deps(all_tasks=tasks,
                rollout=lambda rnd, ts: run_rollout(cfg, rnd, ts, args.appworld_bin, args.experiment),
                judge=judge, tokenizer=trainer.tok, trainer=trainer,
                reload_servers=reload_servers)
    run_online(cfg, deps)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the config**

```yaml
# grpo_online/config/online_v1.yaml
M: 6
G: 8
K: 1
N_rounds: 76
trainer_gpus: "0,1"
gen_gpus: ["2"]
gen_ports: [8101]
adapter_dir: grpo_online/runs/v1/adapter_current
output_dir: grpo_online/runs/v1
rubric_api_model: gpt-4o-mini
rubric_api_key_env: OPENAI_API_KEY
```

- [ ] **Step 3: Write launch script**

```bash
# grpo_online/scripts/launch.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
CFG=grpo_online/config/online_v1.yaml
# 1) seed adapter dir must exist before gen server boots — created by run_online on first start;
#    for a clean boot, run a tiny trainer.save_adapter first OR start the loop, which seeds it,
#    THEN start the gen server. Simplest: start loop in --seed-only mode is out of scope; instead
#    start the gen server pointed at the adapter dir after the loop has written it once.
source /data/minjeong/.envs/grpo9b/bin/activate
# Gen server (GPU 2) — start AFTER adapter_dir exists:
CUDA_VISIBLE_DEVICES=2 python -m grpo_online.gen_server \
  --base-model Qwen/Qwen3.5-9B --adapter grpo_online/runs/v1/adapter_current \
  --served-name Qwen/Qwen3.5-9B --port 8101 > grpo_online/runs/v1/gen.log 2>&1 &
# Loop (GPU 0,1):
CUDA_VISIBLE_DEVICES=0,1 python -m grpo_online.run_online \
  --config $CFG --appworld-bin /data/minjeong/.conda/envs/appworld/bin/appworld
```

- [ ] **Step 4: GPU smoke (manual)**

```
# 1) seed adapter: python -c "from grpo_online.online_trainer import OnlineTrainer; \
#    OnlineTrainer('Qwen/Qwen3.5-9B', [...v3i targets...], device_map='auto').save_adapter('grpo_online/runs/v1/adapter_current')"
# 2) start gen server (GPU2), curl /health → ok, /v1/chat/completions → text
# 3) run loop with N_rounds=2, M=2 (tiny) → expect 2 rounds, each: rollout log, "round N loss=.. kl=.." , /reload_adapter 200
# 4) confirm kl≈0 at round 1 (fresh adapter == gen policy) and no halt
```
Expected: 2 rounds complete; gen server logs two adapter reloads; no OOM/halt.

- [ ] **Step 5: Commit**

```bash
git add grpo_online/run_online.py grpo_online/config/online_v1.yaml grpo_online/scripts/launch.sh
git commit -m "feat(online): entry point + config + launch script"
```

---

## Self-review notes (for the implementer)

- **Spec coverage:** rollout (T6/T8), reward+judge (T3/T4), trainer (T5), sync/loop (T2/T7), config+entry (T1/T8) — all spec sections mapped.
- **Two integration unknowns to verify before T6/T4 real runs** (resolve by reading the code, not guessing):
  1. AppWorld task-subset selection: confirm `ROLLOUT_DATASET` accepts a path to a custom task-id file (else add a config that reads it). Look at `run_round0_rollout_9b.sh` + the rollout jsonnet.
  2. `rubric_reward` prompt builder: confirm/extract `build_ks_baseline_prompt(lm_calls_path) -> messages` and the segment JSON schema. If absent, factor it out of `rubric_reward/run_rollouts.py`.
- **GPU memory lesson:** the trainer's precache-equivalent (old/ref recompute) is per-round and low-mem, but training peak is ~82GB on the lead GPU — keep the gen server off GPUs 0,1.
- **Consistency:** `OnlineConfig` field names used in `run_online.py`/`launch.sh` match Task 1; `score_round` signature `(items, tokenizer, judge_fn)` consistent across T3/T7; `reload_adapter_into` / `train_on_batch` / `save_adapter` names consistent T2/T5/T7.
