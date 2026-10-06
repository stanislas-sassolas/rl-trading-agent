"""A long/flat trading environment (Gymnasium API).

Each step is one hour. The agent sees the market features at the close of hour t plus
its current position, then chooses its position for the next hour:
    action 0 -> fully in cash, action 1 -> fully invested.
Changing position costs `fee` (0.1% = Binance spot taker fee) on the traded amount.

Reward = log growth of the portfolio over the hour, after fees:
    log(1 + position * next_return) + log(1 - fee * |position change|)
Summed over an episode, this is exactly the log of final wealth / initial wealth, so
maximising reward is maximising wealth, with fees and compounding included.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .features import Dataset

REWARD_SCALE = 100.0  # log-returns are ~1e-3 per hour; scaling helps the value network learn


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data: Dataset, fee: float = 0.001, episode_len: int | None = 720,
                 random_start: bool = True, seed: int | None = None):
        super().__init__()
        self.d, self.fee = data, fee
        self.episode_len = episode_len or len(data.X)
        self.random_start = random_start
        n_feat = data.X.shape[1]
        self.observation_space = gym.spaces.Box(-6, 6, shape=(n_feat + 1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.rng = np.random.default_rng(seed)

    def _obs(self):
        return np.append(self.d.X[self.t], np.float32(self.position)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        last_start = len(self.d.X) - self.episode_len
        self.start = int(self.rng.integers(0, max(1, last_start))) if self.random_start else 0
        self.t, self.position, self.steps = self.start, 0, 0
        return self._obs(), {}

    def step(self, action):
        new_pos = int(action)
        cost = self.fee * abs(new_pos - self.position)
        log_growth = np.log1p(new_pos * self.d.next_ret[self.t]) + np.log1p(-cost)
        self.position = new_pos
        self.t += 1
        self.steps += 1
        done = self.steps >= self.episode_len or self.t >= len(self.d.X) - 1
        info = {"log_growth": log_growth, "cost": cost, "position": new_pos}
        return self._obs(), float(log_growth * REWARD_SCALE), done, False, info
