"""Train PPO agents and select the best checkpoint on the validation year.

Training samples random 30-day windows from the training period. Every `eval_every`
steps the policy is run deterministically over the whole validation period; the
checkpoint with the highest validation Sharpe ratio is kept. The test period is not
touched here.

    python -m rltrade.train --asset BTC --steps 1000000 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .backtest import agent_positions, metrics, run_positions
from .env import TradingEnv
from .features import make_splits

ROOT = Path(__file__).resolve().parents[1]
FEES = {"BTC": 0.001, "ETH": 0.001, "GOLD": 0.0005}


class ValidationCallback(BaseCallback):
    def __init__(self, val, fee, eval_every, path):
        super().__init__()
        self.val, self.fee, self.eval_every, self.path = val, fee, eval_every, path
        self.best, self.history = -np.inf, []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every < self.training_env.num_envs:
            m = metrics(run_positions(self.val, agent_positions(self.model, self.val, self.fee), self.fee))
            self.history.append({"steps": self.num_timesteps, **{k: float(v) for k, v in m.items()}})
            if m["Sharpe"] > self.best:
                self.best = m["Sharpe"]
                self.model.save(self.path)
            print(f"  {self.num_timesteps:>8} steps | val Sharpe {m['Sharpe']:+.2f} | return {m['total return']:+.1%} "
                  f"| trades {m['trades']:>4} | invested {m['time invested']:.0%}", flush=True)
        return True


def train(asset: str, steps: int, seed: int, n_envs: int = 8, eval_every: int = 50_000) -> dict:
    data = make_splits(asset)
    fee = FEES[asset]
    venv = DummyVecEnv([lambda i=i: TradingEnv(data["train"], fee=fee, seed=seed * 100 + i) for i in range(n_envs)])
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=1024, batch_size=1024, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2,
                policy_kwargs={"net_arch": [64, 64]}, seed=seed, verbose=0, device="cpu")
    out = ROOT / "models" / asset
    out.mkdir(parents=True, exist_ok=True)
    cb = ValidationCallback(data["val"], fee, eval_every, out / f"ppo_seed{seed}")
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=cb)
    log = {"asset": asset, "seed": seed, "steps": steps, "minutes": round((time.time() - t0) / 60, 1),
           "best_val_sharpe": cb.best, "history": cb.history}
    (out / f"ppo_seed{seed}_log.json").write_text(json.dumps(log, indent=1))
    return log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="BTC", choices=list(FEES))
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-every", type=int, default=50_000)
    args = p.parse_args()
    for s in args.seeds:
        print(f"== {args.asset} seed {s}")
        log = train(args.asset, args.steps, s, eval_every=args.eval_every)
        print(f"   done in {log['minutes']} min, best validation Sharpe {log['best_val_sharpe']:.2f}")


if __name__ == "__main__":
    main()
