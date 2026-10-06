"""Guards against the two classic backtest bugs: look-ahead in features and fee/return mistakes."""

import numpy as np

from rltrade.backtest import buy_and_hold, run_positions
from rltrade.env import TradingEnv
from rltrade.features import compute_features, load_prices, make_splits


def test_features_do_not_use_future_prices():
    df = load_prices("BTC").iloc[:3000]
    full = compute_features(df)
    cut = compute_features(df.iloc[:2000])
    t = cut.index[-2]  # last row whose features are complete in both versions
    cols = [c for c in full.columns if c != "next_ret"]
    np.testing.assert_allclose(full.loc[t, cols].to_numpy(), cut.loc[t, cols].to_numpy(), rtol=1e-9)


def test_scaling_is_fitted_on_train_only():
    s = make_splits("BTC")
    assert abs(s["train"].X.mean(axis=0)).max() < 0.1        # centred on train
    assert abs(s["test"].X[:, 0].mean()) < 1.0                  # test is scaled, not re-centred


def test_env_matches_backtest_for_always_invested():
    d = make_splits("BTC")["val"]
    env = TradingEnv(d, episode_len=None, random_start=False)
    env.reset()
    total, done = 0.0, False
    while not done:
        _, _, done, _, info = env.step(1)
        total += info["log_growth"]
    expected = run_positions(d, buy_and_hold(d))["log_growth"].iloc[:-1].sum()
    assert abs(total - expected) < 1e-9


def test_fees_are_charged_on_each_switch():
    d = make_splits("BTC")["val"]
    pos = np.zeros(len(d.X)); pos[10:20] = 1
    bt = run_positions(d, pos, fee=0.001)
    assert abs(bt["cost"].sum() - 0.002) < 1e-12  # one buy, one sell
