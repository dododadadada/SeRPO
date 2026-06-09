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
