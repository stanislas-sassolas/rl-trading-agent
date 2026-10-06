"""Backtesting: turn a sequence of positions into an equity curve and risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import Dataset


def run_positions(d: Dataset, positions: np.ndarray, fee: float = 0.001) -> pd.DataFrame:
    """positions[t] is held from the close of t to the close of t+1 (0 = cash, 1 = invested)."""
    positions = np.asarray(positions, dtype=float)
    prev = np.concatenate([[0.0], positions[:-1]])
    cost = fee * np.abs(positions - prev)
    log_growth = np.log1p(positions * d.next_ret) + np.log1p(-cost)
    return pd.DataFrame({"position": positions, "cost": cost, "log_growth": log_growth,
                         "equity": np.exp(np.cumsum(log_growth))}, index=d.index)


def metrics(bt: pd.DataFrame) -> dict:
    years = (bt.index[-1] - bt.index[0]).days / 365.25
    per_year = len(bt) / years                         # hours per year actually traded
    r = np.expm1(bt["log_growth"])
    eq = bt["equity"]
    total = eq.iloc[-1] - 1
    vol = r.std() * np.sqrt(per_year)
    return {
        "total return": total,
        "annual return": (1 + total) ** (1 / years) - 1,
        "annual volatility": vol,
        "Sharpe": r.mean() / r.std() * np.sqrt(per_year) if r.std() > 0 else 0.0,
        "max drawdown": (eq / eq.cummax() - 1).min(),
        "trades": int((bt["position"].diff().abs() > 0).sum() + (bt["position"].iloc[0] > 0)),
        "time invested": bt["position"].mean(),
        "fees paid (% of capital)": bt["cost"].sum(),
    }


# ---------- reference strategies ----------

def buy_and_hold(d: Dataset) -> np.ndarray:
    return np.ones(len(d.X))


def sma_crossover(d: Dataset, fast: int = 24, slow: int = 168) -> np.ndarray:
    """Invested when the 24-hour moving average is above the 7-day one (classic trend following)."""
    c = pd.Series(d.close)
    return (c.rolling(fast, min_periods=1).mean() > c.rolling(slow, min_periods=1).mean()).astype(float).to_numpy()


def random_policy(d: Dataset, switch_prob: float, seed: int = 0) -> np.ndarray:
    """Switches position at random with a given hourly probability (matches a strategy's turnover)."""
    rng = np.random.default_rng(seed)
    flips = rng.random(len(d.X)) < switch_prob
    return (np.cumsum(flips) % 2).astype(float)


def agent_positions(model, d: Dataset, fee: float = 0.001) -> np.ndarray:
    """Roll a trained policy over the whole period, deterministically, feeding back its own position."""
    from .env import TradingEnv
    env = TradingEnv(d, fee=fee, episode_len=None, random_start=False)
    obs, _ = env.reset()
    positions, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(action))
        positions.append(info["position"])
    positions.append(positions[-1])  # last row has no next step; keep position for alignment
    return np.array(positions, dtype=float)
