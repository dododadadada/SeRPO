# SeRPO

**Segment-level reward credit assignment for reinforcement learning of multi-step LLM agents.**

An LLM judge segments each agent trajectory and scores the contribution of every
segment; these per-segment scores are used as the advantage signal for GRPO-style
policy optimization. SeRPO is evaluated on the [AppWorld](https://appworld.dev)
benchmark with Qwen3.5-9B.

## Materials

- 📄 **Paper:** [`AI89900_FinalReport_MinjeongBan_SanKwon_final.pdf`](AI89900_FinalReport_MinjeongBan_SanKwon_final.pdf)
- 📊 **Slides:** [`[AI89900]Final_PPT.pdf`](%5BAI89900%5DFinal_PPT.pdf)

## Method overview

- **Segmentation + scoring:** a single LLM-judge call splits a trajectory into
  semantic segments and assigns each a contribution score.
- **Per-segment advantages:** segment scores are z-scored within the rollout group
  and broadcast to the policy's token spans, then optimized with GRPO.
- **Comparisons:** trajectory-level reward (vanilla GRPO), outcome×step credit
  (GiGPO), and segment-collapsed / fail-only ablations.

## Repository layout

| Path | What it is |
|---|---|
| `grpo/` | Offline GRPO: trainer, final v3i configs/scripts (serpo / serpo_avg / vanilla / gigpo), checkpoint eval suite |
| `grpo_online/` | Online GRPO loop: rollout → LLM-judge reward → LoRA update, with smoke/test5/v1 configs |
| `gigpo/` | GiGPO advantage computation (outcome × step-level credit) baseline |
| `rubric_reward/` | LLM-judge rubric scoring (KS variants: baseline / giveup / noerror / shortstep) |
| `viz/` | Method figures and trajectory-comparison visualizations |
| `docs/` | Design docs and plans |

## Quickstart

```bash
git clone https://github.com/dododadadada/SeRPO.git && cd SeRPO
uv sync                                   # trainer env -> .venv (pinned)
cp .env.example .env                      # then fill in OPENAI_API_KEY (rubric judge)
```

- **Offline training:** `grpo/scripts/train_*_v3i.sh` (configs in `grpo/trainer/config/`)
- **Online training (legacy peft backend):** `python -m grpo_online.run_online --config
  grpo_online/config/online_smoke.yaml`, then `online_test5.yaml` → `online_v1.yaml`
- **Online training (vLLM backend, 2 GPUs):** see below — this is the path that is maintained.
- **Eval:** `grpo/eval/eval_*_ckpt.sh` (offline) / `grpo_online/eval_ckpts.py` (online)

## Online GRPO with the vLLM backend

One GPU holds the trainer, the other serves base + the current LoRA through vLLM; the
adapter is hot-swapped after every round, so rollouts always use the current policy.

### Environment (three venvs, all git-ignored)

```bash
uv sync                                                        # .venv        trainer
uv venv .vllm.venv     --python 3.12
uv pip install --python .vllm.venv/bin/python vllm ninja       # .vllm.venv   generation
uv venv .appworld.venv --python 3.11
uv pip install --python .appworld.venv/bin/python \
    "appworld @ git+https://github.com/stonybrooknlp/appworld@42b5bcf" \
    openai'<=1.99.8' litellm jsonnet jinja2 joblib              # .appworld.venv  environment
```

AppWorld harness (the `appworld/` tree is git-ignored; `appworld/experiments/` is the
upstream agents package, `grpo_online/appworld_configs/` holds our configs for it):

```bash
git clone --depth 1 https://github.com/stonybrooknlp/appworld /tmp/aw
cp -r /tmp/aw/experiments appworld/experiments                  # code / configs / prompts
uv pip install --python .appworld.venv/bin/python -e appworld/experiments
cd appworld && ../.appworld.venv/bin/appworld install && ../.appworld.venv/bin/appworld download data
```

Gotchas, all already handled by the launch script or the code:

| Symptom | Cause / fix |
|---|---|
| `apps.bundle is a Git LFS pointer` | pip installs from git leave LFS pointers; fetch the real bundles from `media.githubusercontent.com/media/stonybrooknlp/appworld/<sha>/src/appworld/.source/{apps,tests}.bundle` before `appworld install` |
| `pydantic.errors:ConfigError has been removed in V2` | PyPI `appworld` 0.1.3 pins pydantic<2 and conflicts with litellm — install appworld from git `main` (0.2.0.dev0) |
| `ModuleNotFoundError: appworld_agents` | appworld 0.2.0 resolves configs/prompts from that package — editable-install `appworld/experiments` |
| `'NoneType' object has no attribute 'strip'` | vLLM ≥0.29 returns `reasoning_content: None`; `grpo_online/appworld_patches.py` patches the agent (run by the launch script, idempotent) |
| `ld: cannot find -lcudart` at vLLM boot | flashinfer JIT derives CUDA_HOME from `which nvcc`, which can resolve to an unrelated env; `vllm_gen.build_serve_env` pins `cfg.vllm_cuda_home` and sets `VLLM_USE_FLASHINFER_SAMPLER=0` |
| LoRA "loads" but the policy is unchanged | the trainer names params `model.layers.*` (text-only class) while vLLM expects `model.language_model.layers.*`; vLLM silently ignores unmatched modules. `vllm_gen.export_adapter_for_vllm` remaps the prefixes before loading |
| — | Do **not** merge the LoRA into bf16 weights for serving: at lr 1e-6 the update is below half a bf16 ulp and mostly rounds away |

### Run, resume, validate

```bash
# fresh run (76 rounds; M=6 tasks x G=8 seeds per round, 1 optimizer step per round)
bash grpo_online/scripts/launch_vllm.sh grpo_online/config/online_v2_lr1e5.yaml

# resume: continues from <output_dir>/adapter_current, i.e. the last saved round
bash grpo_online/scripts/launch_vllm.sh grpo_online/config/online_v2_lr1e5.yaml --resume-round <N+1>

# in-run validation: every 4th checkpoint on dev, using the loop's own vLLM server
# (second LoRA slot, so the training adapter is untouched; no extra GPU, no judge calls)
nohup bash grpo_online/scripts/val_during_run.sh grpo_online/config/online_v2_lr1e5.yaml 4 dev &

# after the run: base + every checkpoint on a fixed split, sharded over both GPUs
bash grpo_online/scripts/eval_after_run.sh <run_online_pid> grpo_online/config/online_v2_lr1e5.yaml dev 0 1
```

Configs: `online_v1_vllm.yaml` (lr 1e-6, alpha 32 — the paper's offline v3i recipe) and
`online_v2_lr1e5.yaml` (lr 1e-5, alpha 64). Smoke test first with
`online_smoke_vllm.yaml`; add `--dry-judge` to exercise the whole loop without paying for
the rubric API.

**Resuming on a different machine:** the code above is all that is tracked. Copy the run
directory's `adapter_current/` (or a `ckpt_round_N/`) into `<output_dir>/adapter_current/`
on the new host and launch with `--resume-round N+1`. Only the LoRA weights carry over;
the Adam moments restart, which is immaterial at these learning rates. Checkpoints are
~346 MB (fp32, r=32 over all linear modules), so transfer them directly (`rsync`) rather
than through git.

Large artifacts (checkpoints, rollout outputs, AppWorld data) are intentionally
not tracked — see `.gitignore`.
