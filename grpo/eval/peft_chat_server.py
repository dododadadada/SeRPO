"""Minimal OpenAI-compatible chat-completions server backed by transformers + peft.

Workaround for vLLM 0.18.1 not handling Qwen3.5 + LoRA. AppWorld's eval pipeline
only needs `/v1/chat/completions` over HTTP; vLLM's optimizations aren't required
to verify correctness, just throughput.

Single in-memory model. Concurrent requests serialize through an asyncio lock
so transformers.generate doesn't race on the GPU.

Usage:
  python -m grpo.eval.peft_chat_server \
      --base-model Qwen/Qwen3.5-9B \
      --lora grpo/ckpts/serpo_9b_full_continuous/round1/step_30 \
      --served-name qwen35_9b_serpo_step_30 \
      --port 8003
"""
from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer


_MODEL = None
_TOKENIZER = None
_SERVED_NAME = ""
_LOCK = asyncio.Lock()


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[dict]
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stop: list[str] | str | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    chat_template_kwargs: dict | None = None


def _run_inference(req: ChatRequest) -> dict[str, Any]:
    # Default to non-thinking (matches rollout/eval profile). Some appworld
    # builds reject `extra_body`, so the eval config can't pass
    # enable_thinking; defaulting it here keeps behavior identical without it.
    # A request may still override via top-level chat_template_kwargs.
    ckw = {"enable_thinking": False}
    if req.chat_template_kwargs:
        ckw.update(req.chat_template_kwargs)
    try:
        prompt = _TOKENIZER.apply_chat_template(
            req.messages, tokenize=False, add_generation_prompt=True, **ckw,
        )
    except TypeError:
        # tokenizer's chat template doesn't accept these kwargs → plain render
        prompt = _TOKENIZER.apply_chat_template(
            req.messages, tokenize=False, add_generation_prompt=True,
        )
    inputs = _TOKENIZER(prompt, return_tensors="pt").to(_MODEL.device)
    prompt_len = int(inputs.input_ids.shape[1])
    max_new = int(req.max_completion_tokens or req.max_tokens or 3000)
    temp = float(req.temperature or 0.0)
    do_sample = temp > 0.0

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new,
        "do_sample": do_sample,
        "pad_token_id": _TOKENIZER.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temp
        gen_kwargs["top_p"] = float(req.top_p) if req.top_p is not None else 1.0

    with torch.inference_mode():
        out = _MODEL.generate(**inputs, **gen_kwargs)

    new_tokens = out[0, prompt_len:]
    completion_len = int(new_tokens.size(0))
    text = _TOKENIZER.decode(new_tokens, skip_special_tokens=True)
    finish = "length" if completion_len >= max_new else "stop"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or _SERVED_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_len,
            "total_tokens": prompt_len + completion_len,
        },
    }


app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": _SERVED_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest) -> dict[str, Any]:
    async with _LOCK:
        return await asyncio.get_event_loop().run_in_executor(
            None, _run_inference, req
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--device-map", default="cuda:0")
    args = ap.parse_args()

    global _MODEL, _TOKENIZER, _SERVED_NAME
    _SERVED_NAME = args.served_name
    print(f"loading tokenizer: {args.base_model}")
    _TOKENIZER = AutoTokenizer.from_pretrained(args.base_model)
    if _TOKENIZER.pad_token_id is None:
        _TOKENIZER.pad_token_id = _TOKENIZER.eos_token_id

    print(f"loading base model: {args.base_model} (this takes ~1-2 min)")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map=args.device_map,
    )
    base.eval()

    if args.lora:
        print(f"loading LoRA adapter: {args.lora}")
        _MODEL = PeftModel.from_pretrained(base, args.lora)
    else:
        _MODEL = base
    _MODEL.eval()

    print(f"ready: serving '{_SERVED_NAME}' on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
