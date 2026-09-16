"""Merge a peft LoRA adapter into the base model and save as a full HF model.

Workaround for vLLM 0.18.1 not supporting Qwen3.5 LoRA target modules
(issue #38085). Each merged model is a standalone ~18 GB checkpoint that
vLLM can serve as a base model without --enable-lora.

Usage:
  python -m grpo.eval.merge_lora \
      --base Qwen/Qwen3.5-9B \
      --lora grpo/ckpts/serpo_9b_full_continuous/round1/step_30 \
      --out  grpo/ckpts_merged/serpo_9b_step_30
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(base_name: str, lora_path: Path, out_path: Path) -> None:
    out_path = Path(out_path)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; remove first if you want to overwrite.")
    out_path.mkdir(parents=True)

    print(f"[1/4] loading tokenizer + base ({base_name}) on CPU")
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    base = AutoModelForCausalLM.from_pretrained(
        base_name, dtype=torch.bfloat16, device_map="cpu",
    )

    print(f"[2/4] loading LoRA adapter from {lora_path}")
    model = PeftModel.from_pretrained(base, str(lora_path))

    print("[3/4] merging LoRA into base (peft.merge_and_unload)")
    merged = model.merge_and_unload()

    print(f"[4/4] saving merged model to {out_path}")
    merged.save_pretrained(str(out_path), safe_serialization=True)
    tokenizer.save_pretrained(str(out_path))
    print(f"done. merged size:")
    total = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file())
    print(f"  {total / 1e9:.2f} GB at {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    merge(args.base, args.lora, args.out)


if __name__ == "__main__":
    main()
