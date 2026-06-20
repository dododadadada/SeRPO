# SeRPO

**Segment-level reward credit assignment for reinforcement learning of multi-step LLM agents.**

This repository hosts the paper and presentation materials for **SeRPO**. An LLM
judge segments each agent trajectory and scores the contribution of every segment;
these per-segment scores are used as the advantage signal for GRPO-style policy
optimization. SeRPO is evaluated on the [AppWorld](https://appworld.dev) benchmark
with Qwen3.5-9B.

## Method overview

- **Segmentation + scoring:** a single LLM-judge call splits a trajectory into
  semantic segments and assigns each a contribution score.
- **Per-segment advantages:** segment scores are z-scored within the rollout group
  and broadcast to the policy's token spans, then optimized with GRPO.
- **Comparisons:** trajectory-level reward (vanilla GRPO), outcome×step credit
  (GiGPO), and segment-collapsed / fail-only ablations.

---

_Public materials repository. Research code is maintained separately._
