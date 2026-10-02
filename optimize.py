"""Walk-forward parameter search for a profitable / higher-Sharpe
configuration, across the *whole* project rather than one train/test
split — a follow-up to specs/methodology-fixes-scope.md A-D, at the
user's explicit request to go beyond the mentor-feedback scope and
actively search for a better result.

A single train/test split (what ``main.py`` does) can only tell you
whether *one* parameter choice happened to generalize past *one*
boundary; it says nothing about whether periodically re-optimizing over
time would actually have worked. This script instead walks forward
across several sequential, expanding-window folds
(``quant_project.model_selection.generate_walk_forward_folds``): each fold
re-selects momentum lookback, EMA fast/slow pair, reversal lookback, and
combination method using *only* that fold's own training window, then is
judged on that fold's own held-out test window — never the other way
around. Concatenating every fold's held-out segment end to end gives one
continuous, honestly-out-of-sample return series per strategy: the right
number for "would this have actually worked over time," as opposed to
"did this one split happen to work."

Widening the search like this does *not* remove the risk this project's
whole A-D remediation was about: trying more candidates increases the
chance that something looks good by chance alone (the classic
multiple-comparisons problem), and a single walk-forward run doesn't
correct for that either. Read the per-fold selections printed below, not
just the aggregate number — a configuration that wins by a different,
unstable margin every fold is weaker evidence than one that wins the same
way consistently. And it's entirely possible the honest answer stays "no
edge," the same conclusion notebooks/performance_summary.ipynb reached —
widening the search is a legitimate next step, not a guarantee of finding
something that was never there.
"""

from __future__ import annotations

import main as pipeline
import pandas as pd

from quant_project.audit import log_run
from quant_project.backtest import (
    BacktestResult,
    UnconstrainedBacktester,
    run_multi_strategy_backtest,
)
from quant_project.combination import combine_signals
from quant_project.model_selection import (
    TrainTestSplit,
    generate_walk_forward_folds,
    select_best,
    slice_backtest_result,
)
from quant_project.performance import PerformanceReport, build_performance_report, sharpe_ratio
from quant_project.signals.momentum import ema_crossover_momentum, time_horizon_momentum
from quant_project.signals.reversal import time_horizon_reversal

N_FOLDS = 5
INITIAL_TRAIN_FRACTION = 0.5

MOMENTUM_LOOKBACK_CANDIDATES = (2, 3, 5, 7, 10, 14, 21, 30, 45, 60)
EMA_PAIR_CANDIDATES = (
    (4, 24),
    (8, 24),
    (8, 48),
    (12, 26),
    (12, 48),
    (16, 48),
    (16, 96),
    (24, 72),
    (24, 96),
    (32, 96),
)
REVERSAL_LOOKBACK_CANDIDATES = (1, 2, 3, 5, 7, 10)
"""Wider grids than main.py's — main.py's job is a quick single-split
demo; this script's job is a more thorough (but still not exhaustive)
search. Reversal's lookback is searched here (``main.py`` holds it fixed
at 1 day) since it's exactly the kind of "lookbacks" parameter the
mentor's feedback named and ``main.py`` never tried anything else for."""


def _score_on_train(net_returns: pd.Series, fold: TrainTestSplit) -> float:
    return sharpe_ratio(net_returns.reindex(fold.train_prices.index))


def _select_momentum_lookback(
    prices: pd.DataFrame, returns: pd.DataFrame, fold: TrainTestSplit
) -> int:
    scores = {}
    for lookback in MOMENTUM_LOOKBACK_CANDIDATES:
        signal = time_horizon_momentum(prices, lookback=lookback)
        result = UnconstrainedBacktester(signal, returns).run()
        scores[str(lookback)] = _score_on_train(result.net_returns, fold)
    return int(select_best(scores))


def _select_ema_pair(
    prices: pd.DataFrame, returns: pd.DataFrame, fold: TrainTestSplit
) -> tuple[int, int]:
    scores, pair_by_name = {}, {}
    for fast, slow in EMA_PAIR_CANDIDATES:
        name = f"{fast}-{slow}"
        pair_by_name[name] = (fast, slow)
        signal = ema_crossover_momentum(prices, fast_span=fast, slow_span=slow)
        result = UnconstrainedBacktester(signal, returns).run()
        scores[name] = _score_on_train(result.net_returns, fold)
    return pair_by_name[select_best(scores)]


def _select_reversal_lookback(
    prices: pd.DataFrame, returns: pd.DataFrame, fold: TrainTestSplit
) -> int:
    scores = {}
    for lookback in REVERSAL_LOOKBACK_CANDIDATES:
        signal = time_horizon_reversal(prices, lookback=lookback)
        result = UnconstrainedBacktester(signal, returns).run()
        scores[str(lookback)] = _score_on_train(result.net_returns, fold)
    return int(select_best(scores))


def _select_combination_method(
    signals: dict[str, pd.DataFrame],
    component_results: dict[str, BacktestResult],
    returns: pd.DataFrame,
    fold: TrainTestSplit,
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
        scores[method] = _score_on_train(result.net_returns, fold)
    return select_best(scores)


def _concat_backtest_results(results: list[BacktestResult]) -> BacktestResult:
    return BacktestResult(
        weights=pd.concat([r.weights for r in results]),
        turnover=pd.concat([r.turnover for r in results]),
        gross_returns=pd.concat([r.gross_returns for r in results]),
        net_returns=pd.concat([r.net_returns for r in results]),
    )


def run_walk_forward(
    prices: pd.DataFrame, returns: pd.DataFrame
) -> tuple[dict[str, BacktestResult], list[dict]]:
    """Walk forward across N_FOLDS folds, re-selecting every parameter on
    each fold's own training window, and return (1) every strategy's
    concatenated out-of-sample BacktestResult across all folds and (2) a
    per-fold log of what was selected and how that fold did, for
    transparency beyond the aggregate number."""
    folds = generate_walk_forward_folds(
        prices, n_folds=N_FOLDS, initial_train_fraction=INITIAL_TRAIN_FRACTION
    )

    oos_results: dict[str, list[BacktestResult]] = {
        "momentum": [],
        "ema_crossover": [],
        "reversal": [],
        "combined": [],
    }
    fold_log = []

    for i, fold in enumerate(folds):
        best_lookback = _select_momentum_lookback(prices, returns, fold)
        best_fast, best_slow = _select_ema_pair(prices, returns, fold)
        best_reversal_lookback = _select_reversal_lookback(prices, returns, fold)

        signals = {
            "momentum": time_horizon_momentum(prices, lookback=best_lookback),
            "ema_crossover": ema_crossover_momentum(
                prices, fast_span=best_fast, slow_span=best_slow
            ),
            "reversal": time_horizon_reversal(prices, lookback=best_reversal_lookback),
        }
        results = run_multi_strategy_backtest(signals, returns)

        best_method = _select_combination_method(signals, results, returns, fold)
        combo_kwargs = {
            "equal": {},
            "inverse_vol": {"strategy_returns": {n: r.net_returns for n, r in results.items()}},
            "ic": {"forward_returns": returns},
        }[best_method]
        combined_signal = combine_signals(signals, method=best_method, **combo_kwargs)
        combined_result = run_multi_strategy_backtest({best_method: combined_signal}, returns)[
            best_method
        ]

        fold_oos_sharpe = {}
        for name, result in {**results, "combined": combined_result}.items():
            test_result = slice_backtest_result(result, fold.test_prices.index)
            oos_results[name].append(test_result)
            fold_oos_sharpe[name] = sharpe_ratio(test_result.net_returns)

        fold_log.append(
            {
                "fold": i,
                "test_start": fold.split_date.date(),
                "test_end": fold.test_prices.index[-1].date(),
                "momentum_lookback": best_lookback,
                "ema_pair": (best_fast, best_slow),
                "reversal_lookback": best_reversal_lookback,
                "combo_method": best_method,
                "fold_oos_sharpe": fold_oos_sharpe,
            }
        )

    concatenated = {name: _concat_backtest_results(segs) for name, segs in oos_results.items()}
    return concatenated, fold_log


def _print_report(name: str, report: PerformanceReport) -> None:
    print(
        f"{name:14s}  cum net: {report.cumulative_net.iloc[-1]:+7.2%}"
        f"  ann.ret: {report.annualized_return:+7.2%}"
        f"  ann.vol: {report.annualized_volatility:6.2%}"
        f"  sharpe: {report.sharpe_ratio:+5.2f}"
        f"  max dd: {report.max_drawdown:7.2%}"
        f"  alpha: {report.alpha:+7.2%}"
        f"  alpha t: {report.alpha_t_stat:+5.2f}"
    )


def main() -> None:
    prices = pipeline.load_prices()
    returns = prices.pct_change()
    benchmark_returns = returns[pipeline.BENCHMARK_SYMBOL]

    print(
        f"Walk-forward optimization: {N_FOLDS} folds, "
        f"initial train fraction={INITIAL_TRAIN_FRACTION}"
    )
    print(
        f"Candidate grids: momentum lookback={MOMENTUM_LOOKBACK_CANDIDATES}\n"
        f"                 EMA pairs={EMA_PAIR_CANDIDATES}\n"
        f"                 reversal lookback={REVERSAL_LOOKBACK_CANDIDATES}\n"
    )

    concatenated, fold_log = run_walk_forward(prices, returns)

    print("Per-fold selections (each chosen using ONLY that fold's own training window):")
    for entry in fold_log:
        print(
            f"  fold {entry['fold']}  test {entry['test_start']} -> {entry['test_end']}  "
            f"momentum_lb={entry['momentum_lookback']:>2}  ema={entry['ema_pair']}  "
            f"reversal_lb={entry['reversal_lookback']:>2}  combo={entry['combo_method']:11s}  "
            f"fold OOS sharpe: "
            + "  ".join(
                f"{name}={sharpe:+.2f}" for name, sharpe in entry["fold_oos_sharpe"].items()
            )
        )

    print("\n=== Aggregate walk-forward out-of-sample performance (never in-sample) ===")
    for name, result in concatenated.items():
        report = build_performance_report(result, benchmark_returns=benchmark_returns)
        _print_report(name, report)
        log_run(
            "walk_forward_optimization",
            parameters={
                "strategy": name,
                "n_folds": N_FOLDS,
                "initial_train_fraction": INITIAL_TRAIN_FRACTION,
            },
            result_summary={
                "oos_cumulative_net": report.cumulative_net.iloc[-1],
                "oos_sharpe_ratio": report.sharpe_ratio,
                "oos_alpha_t_stat": report.alpha_t_stat,
            },
        )

    print("\n(audit trail for this run: data/audit_log.jsonl)")


if __name__ == "__main__":
    main()
