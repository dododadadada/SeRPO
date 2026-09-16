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
git clone -b main https://github.com/dododadadada/SeRPO.git && cd SeRPO
uv sync                                   # reproduce the pinned environment
pip install appworld && appworld install  # AppWorld env (installed separately)
appworld download data
cp .env.example .env                      # then fill in OPENAI_API_KEY etc.
```

- **Offline training:** `grpo/scripts/train_*_v3i.sh` (configs in `grpo/trainer/config/`)
- **Online training:** `python -m grpo_online.run_online --config grpo_online/config/online_smoke.yaml`
  to verify plumbing, then `online_test5.yaml` → `online_v1.yaml`
- **Eval:** `grpo/eval/eval_*_ckpt.sh`

Large artifacts (checkpoints, rollout outputs, AppWorld data) are intentionally
not tracked — see `.gitignore`.
