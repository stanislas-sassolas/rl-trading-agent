"""Validation vs. test Sharpe: how much of the validation edge survives on unseen data."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK, ACCENT, MUTED = "#16211E", "#D3263A", "#8A948F"


def main():
    s = json.loads((ROOT / "results" / "summary.json").read_text())
    assets = ["BTC", "ETH", "GOLD"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    for ax, a in zip(axes, assets):
        x = np.arange(2)
        for name, color, off in (("PPO mean of 3 seeds", ACCENT, -0.17), ("Buy & hold", INK, 0.17)):
            vals = [s[f"{a}_val"][name]["Sharpe"], s[f"{a}_test"][name]["Sharpe"]]
            bars = ax.bar(x + off, vals, width=0.32, color=color, label=name.replace(" mean of 3 seeds", " (mean, 3 seeds)"))
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + (0.06 if v >= 0 else -0.2), f"{v:.2f}", ha="center", fontsize=8)
        ax.axhline(0, color=MUTED, lw=0.8)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -0.15), hi * 1.15)
        ax.set_xticks(x, ["validation\n(used to pick the model)", "test\n(unseen)"], fontsize=8)
        ax.set_title(a, loc="left", fontsize=11)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Sharpe ratio (after fees)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "val_vs_test_sharpe.png", dpi=140)


if __name__ == "__main__":
    main()
