"""Unit tests for grpo_online.vllm_gen (no GPU, no vLLM: HTTP is monkeypatched)."""
import os
import json
import pytest
from grpo_online.config import OnlineConfig
from grpo_online.vllm_gen import VllmGenServer, build_serve_cmd


def test_build_serve_cmd_serves_base_under_distinct_name_with_lora():
    cfg = OnlineConfig(gen_backend="vllm", vllm_extra_args=["--language-model-only"])
    cmd = build_serve_cmd(cfg, 8101)
    assert cmd[:3] == [cfg.vllm_bin, "serve", "Qwen/Qwen3.5-9B"]
    assert cmd[cmd.index("--served-model-name") + 1] == cfg.vllm_base_served_name
    assert cfg.vllm_base_served_name != cfg.gen_model_name  # requests must hit the LoRA
    assert "--enable-lora" in cmd
    assert cmd[cmd.index("--max-lora-rank") + 1] == str(cfg.lora_rank)
    assert "--enable-prefix-caching" in cmd
    assert json.loads(cmd[cmd.index("--default-chat-template-kwargs") + 1]) == {"enable_thinking": False}
    assert cmd[-1] == "--language-model-only"


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status; self._p = payload or {}; self.text = json.dumps(self._p)
    def json(self): return self._p
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError(self.status_code)


def _server(tmp_path, prefix_map=None):
    # HTTP-level tests pass an empty prefix map so no adapter files are needed;
    # the export path is covered by test_export_adapter_for_vllm_* below.
    cfg = OnlineConfig(gen_backend="vllm", vllm_lora_key_prefix_map=prefix_map or {})
    return VllmGenServer(cfg, port=8101, gpus="1", log_path=str(tmp_path / "vllm.log"))


def test_load_adapter_unloads_then_loads_same_name(monkeypatch, tmp_path):
    calls = []
    def fake_post(url, json=None, timeout=None):
        calls.append((url.rsplit("/", 1)[-1], json))
        return _Resp(404 if url.endswith("unload_lora_adapter") else 200)
    def fake_get(url, timeout=None):
        return _Resp(200, {"data": [{"id": "qwen35-9b-base"}, {"id": "Qwen/Qwen3.5-9B"}]})
    monkeypatch.setattr("grpo_online.vllm_gen.requests.post", fake_post)
    monkeypatch.setattr("grpo_online.vllm_gen.requests.get", fake_get)
    srv = _server(tmp_path)
    srv.load_adapter(str(tmp_path / "adapter"))
    assert [c[0] for c in calls] == ["unload_lora_adapter", "load_lora_adapter"]
    assert calls[0][1] == {"lora_name": "Qwen/Qwen3.5-9B"}
    assert calls[1][1]["lora_name"] == "Qwen/Qwen3.5-9B"
    assert calls[1][1]["lora_path"] == str(tmp_path / "adapter")  # absolute


def test_load_adapter_raises_when_lora_missing_from_models(monkeypatch, tmp_path):
    monkeypatch.setattr("grpo_online.vllm_gen.requests.post", lambda *a, **k: _Resp(200))
    monkeypatch.setattr("grpo_online.vllm_gen.requests.get",
                        lambda *a, **k: _Resp(200, {"data": [{"id": "qwen35-9b-base"}]}))
    with pytest.raises(RuntimeError, match="not in served models"):
        _server(tmp_path).load_adapter(str(tmp_path / "adapter"))


def test_load_adapter_raises_on_load_error(monkeypatch, tmp_path):
    def fake_post(url, json=None, timeout=None):
        return _Resp(404) if url.endswith("unload_lora_adapter") else _Resp(400, {"error": "bad adapter"})
    monkeypatch.setattr("grpo_online.vllm_gen.requests.post", fake_post)
    with pytest.raises(RuntimeError, match="load_lora_adapter failed"):
        _server(tmp_path).load_adapter(str(tmp_path / "adapter"))


def test_wait_ready_fails_fast_if_process_died(tmp_path):
    class Dead:
        returncode = 1
        def poll(self): return 1
    srv = _server(tmp_path); srv.proc = Dead()
    with pytest.raises(RuntimeError, match="exited with code 1"):
        srv.wait_ready(timeout=60)


def test_export_adapter_for_vllm_remaps_prefixes(tmp_path):
    import torch
    from safetensors.torch import load_file, save_file
    from grpo_online.vllm_gen import export_adapter_for_vllm
    src = tmp_path / "adapter"; src.mkdir()
    (src / "adapter_config.json").write_text('{"r": 32}')
    save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 2),
               "base_model.model.model.layers.3.mlp.up_proj.lora_B.weight": torch.ones(2, 2)},
              str(src / "adapter_model.safetensors"))
    dst = export_adapter_for_vllm(str(src), str(tmp_path / "adapter_vllm"),
                                  {"base_model.model.model.": "base_model.model.model.language_model."})
    sd = load_file(str(tmp_path / "adapter_vllm" / "adapter_model.safetensors"))
    assert sorted(sd) == ["base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
                          "base_model.model.model.language_model.layers.3.mlp.up_proj.lora_B.weight"]
    assert (tmp_path / "adapter_vllm" / "adapter_config.json").read_text() == '{"r": 32}'
    assert dst == str(tmp_path / "adapter_vllm")
    # re-export over an existing dir is fine (per-round overwrite)
    export_adapter_for_vllm(str(src), str(tmp_path / "adapter_vllm"),
                            {"base_model.model.model.": "base_model.model.model.language_model."})


def test_export_adapter_for_vllm_rejects_unmatched_key(tmp_path):
    import torch
    from safetensors.torch import save_file
    from grpo_online.vllm_gen import export_adapter_for_vllm
    src = tmp_path / "adapter"; src.mkdir()
    (src / "adapter_config.json").write_text('{}')
    save_file({"something.else.weight": torch.zeros(1)}, str(src / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="matches 0 prefixes"):
        export_adapter_for_vllm(str(src), str(tmp_path / "out"), {"base_model.": "x."})


def test_load_adapter_exports_remapped_copy(monkeypatch, tmp_path):
    import torch
    from safetensors.torch import save_file
    src = tmp_path / "adapter"; src.mkdir()
    (src / "adapter_config.json").write_text('{}')
    save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(1)},
              str(src / "adapter_model.safetensors"))
    calls = []
    monkeypatch.setattr("grpo_online.vllm_gen.requests.post",
                        lambda url, json=None, timeout=None: (calls.append(json) or _Resp(200)))
    monkeypatch.setattr("grpo_online.vllm_gen.requests.get",
                        lambda *a, **k: _Resp(200, {"data": [{"id": "Qwen/Qwen3.5-9B"}]}))
    _server(tmp_path, {"base_model.model.model.": "base_model.model.model.language_model."}).load_adapter(str(src))
    assert calls[1]["lora_path"] == str(tmp_path / "adapter_vllm")
    assert (tmp_path / "adapter_vllm" / "adapter_model.safetensors").exists()


def test_build_serve_env_pins_cuda_home_and_disables_flashinfer_sampler(tmp_path):
    """flashinfer's JIT links against $CUDA_HOME/lib64; without a pin it follows
    `which nvcc` into whatever env owns it. The sampler is off so no CUDA compile
    happens at boot at all."""
    from grpo_online.vllm_gen import build_serve_env
    cfg = OnlineConfig(gen_backend="vllm", vllm_cuda_home=str(tmp_path))
    env = build_serve_env(cfg, "1", {"PATH": "/usr/bin"})
    assert env["CUDA_HOME"] == str(tmp_path)
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PATH"].startswith(os.path.dirname(os.path.abspath(cfg.vllm_bin)) + ":")
    assert env["PATH"].endswith(":/usr/bin")


def test_build_serve_env_respects_explicit_parent_values(tmp_path):
    from grpo_online.vllm_gen import build_serve_env
    cfg = OnlineConfig(gen_backend="vllm", vllm_cuda_home=str(tmp_path))
    env = build_serve_env(cfg, "0", {"PATH": "/usr/bin", "CUDA_HOME": "/usr/local/cuda-12.8",
                                     "VLLM_USE_FLASHINFER_SAMPLER": "1"})
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "1"
    assert env["CUDA_HOME"] == "/usr/local/cuda-12.8" or "CUDA_HOME" not in env


def test_build_serve_env_omits_missing_cuda_home(tmp_path):
    from grpo_online.vllm_gen import build_serve_env
    cfg = OnlineConfig(gen_backend="vllm", vllm_cuda_home=str(tmp_path / "nope"))
    assert "CUDA_HOME" not in build_serve_env(cfg, "0", {"PATH": "/usr/bin"})
