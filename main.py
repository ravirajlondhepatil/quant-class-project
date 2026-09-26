"""Manual smoke-run of the pipeline built so far: signals -> backtest ->
strategy combination (Features 2-5 in specs/feature-list.md).

Phase 1 (real exchange data ingestion, specs/feature-list.md 1.1-1.5) isn't
built yet, so there's no live OHLCV to run this against. This script
generates a small synthetic multi-asset price panel instead, purely so the
functions that *do* exist can be exercised end to end and inspected by eye.
Swap ``_synthetic_prices()`` out for a real loader once Phase 1 lands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_project.backtest import run_multi_strategy_backtest
from quant_project.combination import combine_signals
from quant_project.signals.momentum import ema_crossover_momentum, time_horizon_momentum
from quant_project.signals.reversal import time_horizon_reversal

ASSETS = ["BTC", "ETH", "SOL", "AVAX"]
N_PERIODS = 250
SEED = 7


def _synthetic_prices() -> pd.DataFrame:
    """Deterministic random-walk daily prices for a handful of assets."""
    rng = np.random.default_rng(SEED)
    daily_returns = rng.normal(loc=0.0003, scale=0.02, size=(N_PERIODS, len(ASSETS)))
    prices = 100 * np.exp(np.cumsum(daily_returns, axis=0))
    index = pd.date_range("2024-01-01", periods=N_PERIODS, freq="D")
    return pd.DataFrame(prices, index=index, columns=ASSETS)


def main() -> None:
    prices = _synthetic_prices()
    returns = prices.pct_change()

    signals = {
        "momentum_7d": time_horizon_momentum(prices, lookback=7),
        "ema_crossover": ema_crossover_momentum(prices, fast_span=8, slow_span=24),
        "reversal_1d": time_horizon_reversal(prices, lookback=1),
    }

    print("=== Individual strategies (Feature 4) ===")
    results = run_multi_strategy_backtest(signals, returns)
    for name, result in results.items():
        cumulative_net = (1 + result.net_returns.fillna(0)).prod() - 1
        avg_turnover = result.turnover.mean()
        print(
            f"{name:15s}  cumulative net return: {cumulative_net:+.2%}"
            f"   avg turnover: {avg_turnover:.2f}"
        )

    print("\n=== Combined strategy (Feature 5) ===")
    for method, kwargs in [
        ("equal", {}),
        ("inverse_vol", {"strategy_returns": {n: r.net_returns for n, r in results.items()}}),
        ("ic", {"forward_returns": returns}),
    ]:
        combined_signal = combine_signals(signals, method=method, **kwargs)
        combined_result = run_multi_strategy_backtest({method: combined_signal}, returns)[method]
        cumulative_net = (1 + combined_result.net_returns.fillna(0)).prod() - 1
        print(f"{method:15s}  cumulative net return: {cumulative_net:+.2%}")


if __name__ == "__main__":
    main()
