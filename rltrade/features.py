"""Market features computed from the past only.

Every feature at hour t uses prices up to and including the close of hour t, never
later. Scaling constants (mean, std) are fitted on the training period only and then
applied unchanged to validation and test, so no statistic leaks from the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

ASSETS = {
    "BTC": "binance-BTCUSD-1h.pkl",
    "ETH": "binance-ETHUSD-1h.pkl",
    "GOLD": "yfinance-GOLDUSD-1h.pkl",
}

# Chronological split: train / validation (model selection) / test (used once)
SPLITS = {
    "BTC": ("2023-12-31 23:00", "2024-12-31 23:00"),
    "ETH": ("2023-12-31 23:00", "2024-12-31 23:00"),
    "GOLD": ("2024-09-30 23:00", "2025-03-31 23:00"),  # shorter history: 2023-11 to 2025-11
}

FEATURES = ["r1", "r4", "r24", "r168", "vol24", "vol168", "rsi14",
            "dist_sma24", "dist_sma168", "volume_z", "hour_sin", "hour_cos"]


def load_prices(asset: str) -> pd.DataFrame:
    df = pd.read_pickle(DATA / ASSETS[asset])[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated()].sort_index().dropna()
    return df[df["close"] > 0]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    logc = np.log(c)
    out = pd.DataFrame(index=df.index)
    for k in (1, 4, 24, 168):
        out[f"r{k}"] = logc.diff(k)                       # log return over the past k hours
    out["vol24"] = out["r1"].rolling(24).std()
    out["vol168"] = out["r1"].rolling(168).std()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi14"] = (100 - 100 / (1 + gain / (loss + 1e-12)) - 50) / 50
    out["dist_sma24"] = c / c.rolling(24).mean() - 1
    out["dist_sma168"] = c / c.rolling(168).mean() - 1
    logv = np.log1p(df["volume"])
    out["volume_z"] = (logv - logv.rolling(168).mean()) / (logv.rolling(168).std() + 1e-9)
    hour = df.index.hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    # return earned by holding from the close of t to the close of t+1: the target, never an input
    out["next_ret"] = c.shift(-1) / c - 1
    return out.replace([np.inf, -np.inf], np.nan).dropna()


@dataclass
class Dataset:
    asset: str
    name: str            # train / val / test
    index: pd.DatetimeIndex
    X: np.ndarray        # scaled features, shape (T, n_features)
    next_ret: np.ndarray # simple return from t to t+1, shape (T,)
    close: np.ndarray


def make_splits(asset: str) -> dict[str, Dataset]:
    feats = compute_features(load_prices(asset))
    prices = load_prices(asset)["close"].reindex(feats.index)
    train_end, val_end = SPLITS[asset]
    parts = {
        "train": feats.loc[:train_end],
        "val": feats.loc[pd.Timestamp(train_end) + pd.Timedelta(hours=1): val_end],
        "test": feats.loc[pd.Timestamp(val_end) + pd.Timedelta(hours=1):],
    }
    mu = parts["train"][FEATURES].mean()
    sd = parts["train"][FEATURES].std().replace(0, 1)
    out = {}
    for name, part in parts.items():
        X = ((part[FEATURES] - mu) / sd).clip(-5, 5).to_numpy(np.float32)
        out[name] = Dataset(asset, name, part.index, X, part["next_ret"].to_numpy(),
                            prices.loc[part.index].to_numpy())
    return out
