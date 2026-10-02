"""Manual smoke-run of the pipeline built so far: real exchange data ->
signals -> backtest -> strategy combination -> performance report
(Features 1-6 in specs/feature-list.md), with every backtest run
audit-logged (Feature 7.4), plus the train/test split and parameter
selection added for specs/methodology-fixes-scope.md C3/C4.

Pulls real daily OHLCV for a 15-coin Binance universe via
``quant_project.data`` (cached locally after the first fetch, so repeat
runs don't re-hit the network). Momentum lookback, EMA fast/slow pair, and
combination method are each chosen by whichever candidate scores best
(net Sharpe) on the *training* period only (``model_selection``); the
chosen configuration is then reported separately in-sample (training) and
out-of-sample (test) — never a single blended number — so a parameter
choice that only worked by coincidence on the training window shows up as
a gap between the two, not as one deceptively good headline figure.

Known remaining issue (out of scope for C, tracked as D in that same scope
doc): this script still just dumps numbers: no stated hypothesis, no
economic rationale, no "here's the conclusion" framing. That's D1's job,
deliberately sequenced after C.
"""

from __future__ import annotations

import pandas as pd

from quant_project.audit import log_run
from quant_project.backtest import (
    BacktestResult,
    UnconstrainedBacktester,
    run_multi_strategy_backtest,
)
from quant_project.combination import combine_signals
from quant_project.data import DataRequest, build_close_price_panel, load_universe_ohlcv
from quant_project.model_selection import select_best, slice_backtest_result, split_prices
from quant_project.performance import PerformanceReport, build_performance_report, sharpe_ratio
from quant_project.signals.momentum import ema_crossover_momentum, time_horizon_momentum
from quant_project.signals.reversal import time_horizon_reversal

EXCHANGE_ID = "binance"
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "AVAX/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "LTC/USDT",
    "ATOM/USDT",
    "UNI/USDT",
    "NEAR/USDT",
    "FIL/USDT",
]
TIMEFRAME = "1d"
START = "2023-01-01"
BENCHMARK_SYMBOL = "BTC/USDT"  # per specs/requirement-spec.md: "e.g. BTC"

TRAIN_FRACTION = 0.7
"""A plain 70/30 split -- specs/methodology-fixes-scope.md C3 didn't
specify a ratio, and nothing about this dataset favors a different one."""

MOMENTUM_LOOKBACK_CANDIDATES = (3, 7, 14, 21)
EMA_PAIR_CANDIDATES = ((8, 24), (12, 26), (16, 48))
"""Candidate grids for parameter selection (C3). Neither is exhaustive --
they're a handful of round, commonly-used choices to select among, not a
claim that these are the only lookbacks/spans worth trying."""


def load_prices() -> pd.DataFrame:
    """Real daily close prices for ``SYMBOLS`` from ``EXCHANGE_ID``, up to
    today — cached locally (``quant_project.data``) after the first fetch."""
    end = pd.Timestamp.now().normalize()
    request = DataRequest(EXCHANGE_ID, SYMBOLS, TIMEFRAME, START, end)
    universe = load_universe_ohlcv(request)
    return build_close_price_panel(universe)


def _score_on_train(net_returns: pd.Series, split) -> float:
    """A candidate's selection score: net Sharpe over the training period
    only (C3) — never the test period, which is what the final choice will
    later be judged on."""
    train_slice = net_returns.reindex(split.train_prices.index)
    return sharpe_ratio(train_slice)


def _select_momentum_lookback(prices: pd.DataFrame, returns: pd.DataFrame, split) -> int:
    scores = {}
    for lookback in MOMENTUM_LOOKBACK_CANDIDATES:
        signal = time_horizon_momentum(prices, lookback=lookback)
        result = UnconstrainedBacktester(signal, returns).run()
        scores[str(lookback)] = _score_on_train(result.net_returns, split)
    return int(select_best(scores))


def _select_ema_pair(prices: pd.DataFrame, returns: pd.DataFrame, split) -> tuple[int, int]:
    scores = {}
    pair_by_name = {}
    for fast, slow in EMA_PAIR_CANDIDATES:
        name = f"{fast}-{slow}"
        pair_by_name[name] = (fast, slow)
        signal = ema_crossover_momentum(prices, fast_span=fast, slow_span=slow)
        result = UnconstrainedBacktester(signal, returns).run()
        scores[name] = _score_on_train(result.net_returns, split)
    return pair_by_name[select_best(scores)]


def _select_combination_method(
    signals: dict[str, pd.DataFrame],
    component_results: dict[str, BacktestResult],
    returns: pd.DataFrame,
    split,
) -> str:
    scores = {}
    for method, kwargs in [
        ("equal", {}),
        (
            "inverse_vol",
            {"strategy_returns": {n: r.net_returns for n, r in component_results.items()}},
        ),
        ("ic", {"forward_returns": returns}),
    ]:
        combined_signal = combine_signals(signals, method=method, **kwargs)
        result = UnconstrainedBacktester(combined_signal, returns).run()
        scores[method] = _score_on_train(result.net_returns, split)
    return select_best(scores)


def _print_report(name: str, report: PerformanceReport) -> None:
    print(
        f"{name:22s}  cum net: {report.cumulative_net.iloc[-1]:+7.2%}"
        f"  ann.ret: {report.annualized_return:+7.2%}"
        f"  ann.vol: {report.annualized_volatility:6.2%}"
        f"  sharpe: {report.sharpe_ratio:+5.2f}"
        f"  max dd: {report.max_drawdown:7.2%}"
        f"  alpha: {report.alpha:+7.2%}"
        f"  beta: {report.beta:+5.2f}"
        f"  alpha t: {report.alpha_t_stat:+5.2f}"
        f"  corr: {report.correlation:+5.2f}"
    )


def _print_in_sample_out_of_sample(
    name: str, result: BacktestResult, split, benchmark_returns: pd.Series
) -> None:
    train_result = slice_backtest_result(result, split.train_prices.index)
    test_result = slice_backtest_result(result, split.test_prices.index)
    train_report = build_performance_report(train_result, benchmark_returns=benchmark_returns)
    test_report = build_performance_report(test_result, benchmark_returns=benchmark_returns)
    _print_report(f"{name} (train)", train_report)
    _print_report(f"{name} (test)", test_report)
    log_run(
        "backtest",
        parameters={"strategy": name},
        result_summary={
            "train_cumulative_net": train_report.cumulative_net.iloc[-1],
            "train_sharpe_ratio": train_report.sharpe_ratio,
            "test_cumulative_net": test_report.cumulative_net.iloc[-1],
            "test_sharpe_ratio": test_report.sharpe_ratio,
        },
    )


def main() -> None:
    prices = load_prices()
    returns = prices.pct_change()
    benchmark_returns = returns[BENCHMARK_SYMBOL]

    split = split_prices(prices, train_fraction=TRAIN_FRACTION)
    print(
        f"Train/test split: train {split.train_prices.index[0].date()} -> "
        f"{split.train_prices.index[-1].date()} ({len(split.train_prices)} bars), "
        f"test {split.split_date.date()} -> {prices.index[-1].date()} "
        f"({len(split.test_prices)} bars)\n"
    )

    best_lookback = _select_momentum_lookback(prices, returns, split)
    best_fast, best_slow = _select_ema_pair(prices, returns, split)
    print(
        f"Selected on training data: momentum lookback={best_lookback}d, "
        f"EMA pair=({best_fast}, {best_slow})\n"
    )

    signals = {
        "momentum": time_horizon_momentum(prices, lookback=best_lookback),
        "ema_crossover": ema_crossover_momentum(prices, fast_span=best_fast, slow_span=best_slow),
        "reversal_1d": time_horizon_reversal(prices, lookback=1),
    }

    print("=== Individual strategies: in-sample (train) vs. out-of-sample (test) ===")
    results = run_multi_strategy_backtest(signals, returns)
    for name, result in results.items():
        _print_in_sample_out_of_sample(name, result, split, benchmark_returns)

    best_method = _select_combination_method(signals, results, returns, split)
    print(f"\nSelected combination method on training data: {best_method}\n")

    combo_kwargs = {
        "equal": {},
        "inverse_vol": {"strategy_returns": {n: r.net_returns for n, r in results.items()}},
        "ic": {"forward_returns": returns},
    }[best_method]
    combined_signal = combine_signals(signals, method=best_method, **combo_kwargs)
    combined_result = run_multi_strategy_backtest({best_method: combined_signal}, returns)[
        best_method
    ]

    print(f"=== Combined strategy ({best_method}): in-sample (train) vs. out-of-sample (test) ===")
    _print_in_sample_out_of_sample(
        f"combined_{best_method}", combined_result, split, benchmark_returns
    )

    print("\n(audit trail for this run: data/audit_log.jsonl)")


if __name__ == "__main__":
    main()
