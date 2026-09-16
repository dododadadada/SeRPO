"""Graft a text-only merged Qwen3.5 checkpoint back into the full VLM weight set
so vLLM 0.18.1 (which only knows the `qwen3_5` VLM arch, not the `qwen3_5_text`
text-only arch emitted by transformers 5.x) can serve it natively.

merged text weights (427 keys, with step_50 deltas, exact hub key names)
  + hub vision weights (348 keys vLLM expects)  -> full 775-key VLM checkpoint
  + hub config.json / tokenizer.
"""
from __future__ import annotations
import json, glob, shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SNAP = "/data/minjeong/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MERGED = "grpo/ckpts_merged/serpo_9b_v3i_step50/model.safetensors"
OUT = Path("grpo/ckpts_merged/serpo_9b_v3i_step50_vlm")

OUT.mkdir(parents=True, exist_ok=True)
tensors: dict[str, torch.Tensor] = {}

print("[1/4] loading merged text weights")
with safe_open(MERGED, "pt") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
merged_keys = set(tensors)
print(f"      {len(merged_keys)} text keys")

print("[2/4] grafting hub vision weights (keys not in merged)")
n_graft = 0
for shard in sorted(glob.glob(SNAP + "/*.safetensors")):
    with safe_open(shard, "pt") as f:
        for k in f.keys():
            if k not in tensors:
                tensors[k] = f.get_tensor(k)
                n_graft += 1
print(f"      grafted {n_graft} vision keys -> total {len(tensors)}")

print("[3/4] saving combined safetensors (single file)")
save_file(tensors, str(OUT / "model.safetensors"), metadata={"format": "pt"})

print("[4/4] copying hub config + tokenizer")
for fn in ("config.json", "generation_config.json", "tokenizer.json",
           "tokenizer_config.json", "vocab.json", "merges.txt",
           "preprocessor_config.json", "chat_template.jinja"):
    src = Path(SNAP) / fn
    if src.exists():
        shutil.copy(src, OUT / fn)
        print("      +", fn)
print("done:", OUT)
