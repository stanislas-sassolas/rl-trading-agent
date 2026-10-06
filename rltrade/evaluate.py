"""Final evaluation on the test period (run once, after training and model selection).

For each asset: the three PPO agents (one per seed, best validation checkpoint) and
four reference strategies, all with the same fees:
  - buy & hold
  - SMA crossover (24 h vs 7 days)
  - random switching with the same turnover as the agents (is there skill beyond luck?)
  - cash (0%)

    python -m rltrade.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .backtest import agent_positions, buy_and_hold, metrics, random_policy, run_positions, sma_crossover
from .features import make_splits
from .train import FEES

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
ASSETS = ("BTC", "ETH", "GOLD")
INK, MUTED, ACCENT, ALT = "#16211E", "#8A948F", "#D3263A", "#2C6E8E"


def evaluate(asset: str, split: str = "test"):
    d = make_splits(asset)[split]
    fee = FEES[asset]
    curves, rows = {}, {}
    agent_bts = []
    for s in SEEDS:
        model = PPO.load(ROOT / "models" / asset / f"ppo_seed{s}", device="cpu")
        bt = run_positions(d, agent_positions(model, d, fee), fee)
        agent_bts.append(bt)
        rows[f"PPO seed {s}"] = metrics(bt)
        curves[f"PPO seed {s}"] = bt["equity"]
    turnover = np.mean([m["trades"] for k, m in rows.items()]) / len(d.X)
    for name, pos in (("Buy & hold", buy_and_hold(d)), ("SMA crossover 24h/7d", sma_crossover(d)),
                      ("Random, same turnover", random_policy(d, turnover, seed=1)),
                      ("Cash", np.zeros(len(d.X)))):
        bt = run_positions(d, pos, fee)
        rows[name] = metrics(bt)
        curves[name] = bt["equity"]
    table = pd.DataFrame(rows).T
    agents = table.loc[[f"PPO seed {s}" for s in SEEDS]]
    table.loc["PPO mean of 3 seeds"] = agents.mean()
    return table, curves, agent_bts


def plot(asset_curves: dict, path: Path):
    fig, axes = plt.subplots(1, len(asset_curves), figsize=(5.2 * len(asset_curves), 3.8), squeeze=False)
    for ax, (asset, (curves, bts)) in zip(axes[0], asset_curves.items()):
        for s in SEEDS:
            ax.plot(curves[f"PPO seed {s}"], color=ACCENT, lw=1.1, alpha=0.85, label="PPO (3 seeds)" if s == 0 else None)
        ax.plot(curves["Buy & hold"], color=INK, lw=1.6, label="Buy & hold")
        ax.plot(curves["SMA crossover 24h/7d"], color=ALT, lw=1.2, ls="--", label="SMA crossover")
        ax.axhline(1, color=MUTED, lw=0.8)
        ax.set_title(f"{asset} · test period (unseen)", fontsize=11, loc="left")
        ax.set_ylabel("Growth of 1 (after fees)")
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_positions(asset: str, curves, bts, path: Path):
    """Price with the periods where one agent was invested shaded, to see what it learned."""
    d = make_splits(asset)["test"]
    bt = bts[0]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(d.index, d.close, color=INK, lw=1)
    inv = bt["position"].to_numpy() > 0
    ax.fill_between(d.index, d.close.min(), d.close.max(), where=inv, color=ACCENT, alpha=0.12, step="post",
                    label="agent invested (seed 0)")
    ax.set_title(f"{asset} price, test period: when is the agent in the market?", fontsize=11, loc="left")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    out_fig = ROOT / "figures"
    out_fig.mkdir(exist_ok=True)
    out_res = ROOT / "results"
    summary, all_curves = {}, {}
    for asset in ASSETS:
        for split in ("val", "test"):
            table, curves, bts = evaluate(asset, split)
            fmt = table.copy()
            for c in ("total return", "annual return", "annual volatility", "max drawdown", "time invested",
                      "fees paid (% of capital)"):
                fmt[c] = (fmt[c] * 100).round(1)
            fmt["Sharpe"] = fmt["Sharpe"].round(2)
            fmt["trades"] = fmt["trades"].round(0).astype(int)
            fmt = fmt.rename(columns=lambda c: c + (" %" if c not in ("Sharpe", "trades") and "%" not in c else ""))
            (out_res / f"{asset}_{split}.md").write_text(fmt.to_markdown(), encoding="utf-8")
            summary[f"{asset}_{split}"] = table.to_dict(orient="index")
            print(f"\n===== {asset} · {split}\n{fmt.to_string()}")
            if split == "test":
                all_curves[asset] = (curves, bts)
                plot_positions(asset, curves, bts, out_fig / f"{asset}_positions_test.png")
    plot(all_curves, out_fig / "equity_test.png")
    (out_res / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
