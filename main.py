"""Manual smoke-run of the pipeline built so far: signals -> backtest ->
strategy combination -> performance report (Features 2-6 in
specs/feature-list.md), with every backtest run audit-logged (Feature 7.4).

Phase 1's real exchange data ingestion (specs/feature-list.md 1.1-1.5,
src/quant_project/data.py) needs network access this script deliberately
doesn't require, so it isn't wired in here — see
tests/test_quant_project.py for that module's (network-free) tests against
a fake exchange client. This script generates a small synthetic
multi-asset price panel instead, purely so the functions that *do* run
without a network call can be exercised end to end and inspected by eye.
Swap ``_synthetic_prices()`` out for ``quant_project.data.load_universe_ohlcv``
once you're ready to point this at a real exchange.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_project.audit import log_run
from quant_project.backtest import run_multi_strategy_backtest
from quant_project.combination import combine_signals
from quant_project.performance import build_performance_report
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


def _print_report(name: str, report) -> None:
    print(
        f"{name:15s}  cum net: {report.cumulative_net.iloc[-1]:+7.2%}"
        f"  ann.ret: {report.annualized_return:+7.2%}"
        f"  ann.vol: {report.annualized_volatility:6.2%}"
        f"  sharpe: {report.sharpe_ratio:+5.2f}"
        f"  max dd: {report.max_drawdown:7.2%}"
        f"  alpha: {report.alpha:+7.2%}"
        f"  beta: {report.beta:+5.2f}"
    )


def main() -> None:
    prices = _synthetic_prices()
    returns = prices.pct_change()
    benchmark_returns = returns["BTC"]  # per specs/requirement-spec.md: "e.g. BTC"

    signals = {
        "momentum_7d": time_horizon_momentum(prices, lookback=7),
        "ema_crossover": ema_crossover_momentum(prices, fast_span=8, slow_span=24),
        "reversal_1d": time_horizon_reversal(prices, lookback=1),
    }

    print("=== Individual strategies (Feature 4) — performance report (Feature 6) ===")
    results = run_multi_strategy_backtest(signals, returns)
    for name, result in results.items():
        report = build_performance_report(result, benchmark_returns=benchmark_returns)
        _print_report(name, report)
        log_run(
            "backtest",
            parameters={"strategy": name, "assets": ASSETS, "n_periods": N_PERIODS},
            result_summary={
                "cumulative_net": report.cumulative_net.iloc[-1],
                "sharpe_ratio": report.sharpe_ratio,
                "max_drawdown": report.max_drawdown,
            },
        )

    print("\n=== Combined strategy (Feature 5) — performance report (Feature 6) ===")
    for method, kwargs in [
        ("equal", {}),
        ("inverse_vol", {"strategy_returns": {n: r.net_returns for n, r in results.items()}}),
        ("ic", {"forward_returns": returns}),
    ]:
        combined_signal = combine_signals(signals, method=method, **kwargs)
        combined_result = run_multi_strategy_backtest({method: combined_signal}, returns)[method]
        report = build_performance_report(combined_result, benchmark_returns=benchmark_returns)
        _print_report(method, report)
        log_run(
            "backtest",
            parameters={
                "strategy": f"combined_{method}",
                "combination_method": method,
                "component_strategies": list(signals.keys()),
            },
            result_summary={
                "cumulative_net": report.cumulative_net.iloc[-1],
                "sharpe_ratio": report.sharpe_ratio,
                "max_drawdown": report.max_drawdown,
            },
        )

    print("\n(audit trail for this run: data/audit_log.jsonl)")


if __name__ == "__main__":
    main()
