"""vLLM generation backend for online GRPO: base served once, LoRA hot-swapped per round.

Replaces gen_server.py (transformers.generate behind an asyncio lock => one request
at a time, full re-prefill every step). vLLM gives continuous batching + prefix
caching, so the M x G rollout can run with many concurrent AppWorld workers.

Why LoRA-in-vLLM rather than merge-and-serve: at lr=1e-6 the LoRA delta is
|dW| ~ 1e-6..1e-5, below half a bf16 ulp of the base weights (~6e-5 at |W|~0.02).
Merging into bf16 rounds most of the update away (measured on Qwen3.5-9B: 5-16% of
||dW|| survives in early rounds). vLLM's LoRA path keeps A/B separate and adds
their product to the activations, so the delta is preserved like in peft.

Sync protocol per round (load_adapter): unload the LoRA named `lora_name`, then load
adapter_dir under the same name. vLLM assigns a fresh lora_int_id on every load and
its worker cache is keyed by that id, so the weights are re-read from disk.

AppWorld addresses the policy by model name, so the LoRA is registered under
cfg.gen_model_name (what AppWorld requests) and the *base* is served under
cfg.vllm_base_served_name.

Adapter key naming: the trainer loads Qwen3.5-9B through AutoModelForCausalLM,
i.e. the text-only Qwen3_5ForCausalLM whose params are `model.layers.*`, while the
checkpoint (and vLLM's Qwen3_5ForConditionalGeneration) name them
`model.language_model.layers.*`. vLLM silently ignores LoRA tensors whose module
name matches nothing (verified: adapter "loaded", logprobs identical to base), so
load_adapter first writes a copy with the prefixes remapped
(cfg.vllm_lora_key_prefix_map) and loads that. The trainer's own adapter dir stays
plain peft format, eval-compatible with peft_chat_server.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import time

import requests

logger = logging.getLogger("online")


def build_serve_cmd(cfg, port: int) -> list[str]:
    cmd = [
        cfg.vllm_bin, "serve", cfg.model_name,
        "--host", "0.0.0.0", "--port", str(port),
        "--served-model-name", cfg.vllm_base_served_name,
        "--dtype", "bfloat16",
        "--max-model-len", str(cfg.vllm_max_model_len),
        "--max-num-seqs", str(cfg.vllm_max_num_seqs),
        "--gpu-memory-utilization", str(cfg.vllm_gpu_mem_util),
        "--enable-lora", "--max-lora-rank", str(cfg.lora_rank), "--max-loras", "2",
        "--enable-prefix-caching",
        # Same default as gen_server.py (enable_thinking=False); tokenize_trajectory
        # assumes the empty <think> block this renders before the assistant turn.
        "--default-chat-template-kwargs", json.dumps({"enable_thinking": False}),
    ]
    cmd += list(getattr(cfg, "vllm_extra_args", []) or [])
    return cmd


def export_adapter_for_vllm(src_dir: str, dst_dir: str, prefix_map: dict[str, str]) -> str:
    """Copy a peft adapter dir to dst_dir with tensor-name prefixes remapped for
    vLLM's module naming. Every tensor must match exactly one prefix in
    prefix_map (a no-op identity map is allowed); anything else is an error rather
    than a silent skip. Returns dst_dir."""
    from safetensors.torch import load_file, save_file
    src, dst = os.path.abspath(src_dir), os.path.abspath(dst_dir)
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    shutil.copy(os.path.join(src, "adapter_config.json"), tmp)
    sd = load_file(os.path.join(src, "adapter_model.safetensors"))
    out = {}
    for k, v in sd.items():
        hits = [p for p in prefix_map if k.startswith(p)]
        if len(hits) != 1:
            raise ValueError(f"adapter key {k!r} matches {len(hits)} prefixes in {list(prefix_map)}")
        out[prefix_map[hits[0]] + k[len(hits[0]):]] = v
    save_file(out, os.path.join(tmp, "adapter_model.safetensors"), metadata={"format": "pt"})
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.rename(tmp, dst)
    return dst


def build_serve_env(cfg, gpus: str, base_env) -> dict[str, str]:
    """Environment overrides for the `vllm serve` child process.

    - PATH: vllm_bin lives in its own venv; put that bin dir first so JIT helpers
      it shells out to (flashinfer -> `ninja`) resolve without activation.
    - CUDA_HOME: flashinfer's JIT links against `$CUDA_HOME/lib64` and derives
      CUDA_HOME from `which nvcc` when unset. On a shared box that can resolve to
      an unrelated env whose lib64 has no `libcudart.so`, and the build dies with
      "cannot find -lcudart". Pin a real toolkit (cfg.vllm_cuda_home) when it
      exists; an explicit CUDA_HOME in the parent env wins.
    - VLLM_USE_FLASHINFER_SAMPLER: off by default, so sampling uses vLLM's
      PyTorch/Triton path — no boot-time CUDA compile, and the same sampler the
      run started with.
    """
    bin_dir = os.path.dirname(os.path.abspath(cfg.vllm_bin))
    env = {
        "PATH": bin_dir + os.pathsep + base_env.get("PATH", ""),
        "CUDA_VISIBLE_DEVICES": gpus,
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        "VLLM_USE_FLASHINFER_SAMPLER": base_env.get("VLLM_USE_FLASHINFER_SAMPLER", "0"),
    }
    cuda_home = base_env.get("CUDA_HOME") or getattr(cfg, "vllm_cuda_home", "")
    if cuda_home and os.path.isdir(cuda_home):
        env["CUDA_HOME"] = cuda_home
    return env


class VllmGenServer:
    def __init__(self, cfg, port: int, gpus: str, log_path: str, lora_name: str | None = None):
        self.cfg = cfg
        self.port = port
        self.gpus = gpus
        self.log_path = log_path
        # Default: the name AppWorld requests during rollout. An in-run evaluator
        # attaches to the SAME server with a different name so it occupies the
        # second LoRA slot (--max-loras 2) and never disturbs the training loop's
        # adapter (see eval_ckpts --attach-port).
        self.lora_name = lora_name or cfg.gen_model_name
        self.base = f"http://localhost:{port}"
        self.proc: subprocess.Popen | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self, ready_timeout: float = 1800.0) -> None:
        cfg = self.cfg
        cmd = build_serve_cmd(cfg, self.port)
        env = {**os.environ, **build_serve_env(cfg, self.gpus, os.environ)}
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        logger.info("vllm: starting on GPU(s) %s port %d: %s", self.gpus, self.port, " ".join(cmd))
        # New session => we can kill the whole vLLM process tree (engine core,
        # workers) by process group without touching anything else.
        self.proc = subprocess.Popen(
            cmd, env=env, start_new_session=True,
            stdout=open(self.log_path, "ab"), stderr=subprocess.STDOUT)
        self.wait_ready(ready_timeout)

    def wait_ready(self, timeout: float) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"vllm exited with code {self.proc.returncode} before becoming "
                    f"ready; see {self.log_path}")
            try:
                if requests.get(f"{self.base}/health", timeout=5).status_code == 200:
                    logger.info("vllm: ready after %.0fs", time.time() - t0)
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        raise TimeoutError(f"vllm not ready after {timeout:.0f}s; see {self.log_path}")

    def stop(self, grace: float = 30.0) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        pgid = os.getpgid(self.proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            self.proc.wait(timeout=30)
        logger.info("vllm: stopped (pgid %d)", pgid)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    # -- adapter sync ------------------------------------------------------
    def served_models(self) -> list[str]:
        r = requests.get(f"{self.base}/v1/models", timeout=30)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    def load_adapter(self, adapter_dir: str) -> None:
        """Hot-swap the served LoRA to the weights at adapter_dir (same name)."""
        t0 = time.time()
        pm = getattr(self.cfg, "vllm_lora_key_prefix_map", None) or {}
        if pm:
            path = export_adapter_for_vllm(adapter_dir, os.path.abspath(adapter_dir).rstrip("/") + "_vllm", pm)
        else:
            path = os.path.abspath(adapter_dir)
        # Unload the previous round's adapter under this name (404/400 on first load).
        r = requests.post(f"{self.base}/v1/unload_lora_adapter",
                          json={"lora_name": self.lora_name}, timeout=120)
        if r.status_code not in (200, 400, 404):
            r.raise_for_status()
        r = requests.post(f"{self.base}/v1/load_lora_adapter",
                          json={"lora_name": self.lora_name, "lora_path": path}, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(f"vllm load_lora_adapter failed ({r.status_code}): {r.text[:500]}")
        models = self.served_models()
        if self.lora_name not in models:
            raise RuntimeError(f"vllm: LoRA {self.lora_name!r} not in served models {models}")
        logger.info("vllm: adapter reloaded from %s in %.1fs", path, time.time() - t0)
