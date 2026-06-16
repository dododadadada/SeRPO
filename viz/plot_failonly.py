"""Fail-Only advantage-distribution plot: GRPO/GiGPO vs SeRPO.

On the fail-only batch every trajectory has the same outcome (all failed), so a
group-normalized OUTCOME advantage (Vanilla GRPO, GiGPO) is exactly 0 for every
token -> zero gradient -> no learning. SeRPO's SEGMENT-level advantages vary
within each trajectory -> nonzero gradient. This plot shows both distributions
on the same real batch.

Data: grpo/archive/data/round0_ks_baseline/serpo_failonly_binary.parquet
Outputs: viz/failonly_advantages.png (+ .pdf)

Usage: .venv/bin/python3 viz/plot_failonly.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "grpo/archive/data/round0_ks_baseline/serpo_failonly_binary.parquet"
OUT_PNG = ROOT / "viz/failonly_advantages.png"
OUT_PDF = ROOT / "viz/failonly_advantages.pdf"


def collect_advantages(df: pd.DataFrame) -> np.ndarray:
    """SeRPO per-token advantages over response tokens only (where the mask is on)."""
    vals = []
    for adv, mask in zip(df["advantages"], df["response_mask"]):
        a = np.asarray(adv, dtype=float)
        m = np.asarray(mask, dtype=bool)
        n = min(len(a), len(m))
        vals.append(a[:n][m[:n]])
    return np.concatenate(vals) if vals else np.array([])


def main() -> None:
    df = pd.read_parquet(PARQUET)
    n_tasks = df["task_id"].nunique()
    n_rows = len(df)

    serpo = collect_advantages(df)
    # GRPO/GiGPO: group-normalize the scalar outcome per task. All outcomes equal
    # within (and across) groups -> std 0 -> advantage 0 for every token.
    grp_std = df.groupby("task_id")["outcome"].transform(lambda s: np.std(s.astype(float)))
    # Every group std is 0 here, so the normalized outcome advantage is 0 everywhere.
    grpo_adv = np.zeros_like(serpo)  # one value per SeRPO token, all exactly 0

    serpo_nonzero_frac = float((serpo != 0).mean())
    serpo_std = float(serpo.std())

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=200)

    # SeRPO distribution (real spread).
    ax.hist(serpo, bins=60, color="#22c55e", alpha=0.75,
            label=f"SeRPO  (segment-level)\nstd={serpo_std:.2f}, {serpo_nonzero_frac*100:.0f}% tokens nonzero")
    # GRPO/GiGPO: a single spike at 0. Draw as a bold vertical line + marker so it
    # reads as "all mass at exactly 0" rather than getting lost in the histogram.
    ax.axvline(0.0, color="#ef4444", lw=3, label="Vanilla GRPO / GiGPO  (outcome-level)\nstd=0  →  gradient = 0")
    ax.annotate("all advantages = 0\n(no gradient)", xy=(0, 0), xytext=(0.30, 0.82),
                textcoords="axes fraction", color="#b91c1c", fontsize=10, fontweight="bold",
                ha="left", arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.5))

    ax.set_xlabel("advantage value (per response token)")
    ax.set_ylabel("token count")
    ax.set_title(f"Fail-Only batch: advantage distribution\n"
                 f"{n_tasks} fail-only tasks × {n_rows // n_tasks} seeds "
                 f"({n_rows} trajectories, every outcome = fail)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print(f"tasks={n_tasks} trajectories={n_rows}")
    print(f"SeRPO advantages: std={serpo_std:.3f} nonzero_frac={serpo_nonzero_frac:.3f} "
          f"range=[{serpo.min():.2f},{serpo.max():.2f}]")
    print(f"GRPO/GiGPO advantages: all {len(grpo_adv)} tokens = 0 (group outcome variance 0)")
    print(f"Wrote {OUT_PNG} and {OUT_PDF}")


if __name__ == "__main__":
    main()
