"""How much of this could be luck? Error bars and a multiple-testing check.

Two questions any interviewer asks about a backtest, answered with numbers:

1. How precise is each Sharpe ratio? Every strategy in the README table
   gets a 95% confidence interval, before and after costs, in both
   periods (``performance.sharpe_confidence_interval``, a block bootstrap).
2. How many strategies were tried before the best one turned up? The trial
   register lists every configuration the project's scripts have
   backtested, estimates how many independent bets they amount to, and
   runs the best one through the deflated Sharpe ratio test (Bailey &
   López de Prado, 2014): the probability that its true Sharpe beats what
   the best of that many no-edge strategies would show by luck.

The notebook calls these functions for its "How much of this could be
luck?" section, so it shows the numbers this script prints.

Run: uv run python overfitting_check.py
"""

from __future__ import annotations

import main as pipeline
import optimize
import pandas as pd
import short_horizon_study as study

from quant_project.audit import log_run
from quant_project.backtest import (
    BacktestResult,
    UnconstrainedBacktester,
    run_multi_strategy_backtest,
)
from quant_project.combination import combine_signals
from quant_project.model_selection import slice_backtest_result
from quant_project.performance import (
    deflated_sharpe_ratio,
    effective_number_of_trials,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_confidence_interval,
    sharpe_ratio,
)
from quant_project.signals.momentum import ema_crossover_momentum, time_horizon_momentum
from quant_project.signals.reversal import time_horizon_reversal

COMMON_START_BAR = 100
"""Trials are compared from the 101st bar on. By then every configuration
has a live signal (the slowest EMA pair needs 96 bars), so none of them is
credited with a quiet warm-up period the others didn't get."""


def _main_components(data: study.StudyData) -> tuple[dict, dict, str, dict]:
    """main.py's training-period selections, made the same way main.py makes
    them: signals, their backtests, the chosen combination method, and the
    parameters behind the labels."""
    prices, returns, split = data.prices, data.returns, data.split
    lookback = pipeline._select_momentum_lookback(prices, returns, split)
    fast, slow = pipeline._select_ema_pair(prices, returns, split)
    signals = {
        "momentum": time_horizon_momentum(prices, lookback=lookback),
        "ema_crossover": ema_crossover_momentum(prices, fast_span=fast, slow_span=slow),
        "reversal_1d": time_horizon_reversal(prices, lookback=1),
    }
    results = run_multi_strategy_backtest(signals, returns)
    method = pipeline._select_combination_method(signals, results, returns, split)
    params = {"lookback": lookback, "fast": fast, "slow": slow}
    return signals, results, method, params


def _combination_kwargs(method: str, results: dict, returns: pd.DataFrame) -> dict:
    return {
        "equal": {},
        "inverse_vol": {"strategy_returns": {n: r.net_returns for n, r in results.items()}},
        "ic": {"forward_returns": returns},
    }[method]


def selected_strategies(data: study.StudyData) -> dict[str, BacktestResult]:
    """The five strategies in the README table: main.py's three selected
    signals and its selected combination, plus the flipped 1-day trade."""
    signals, results, method, params = _main_components(data)
    combined_signal = combine_signals(
        signals, method=method, **_combination_kwargs(method, results, data.returns)
    )
    return {
        f"{params['lookback']}-day momentum": results["momentum"],
        f"EMA crossover ({params['fast']}/{params['slow']})": results["ema_crossover"],
        "1-day reversal": results["reversal_1d"],
        "1-day momentum (reversal flipped)": UnconstrainedBacktester(
            time_horizon_momentum(data.prices, lookback=1), data.returns
        ).run(),
        f"Combined ({method.replace('_', '-')} weights)": UnconstrainedBacktester(
            combined_signal, data.returns
        ).run(),
    }


def interval_table(data: study.StudyData) -> pd.DataFrame:
    """Sharpe ratio and its 95% bootstrap interval, before and after costs,
    for each strategy in each period."""
    rows = {}
    for name, result in selected_strategies(data).items():
        for period, index in data.periods.items():
            sliced = slice_backtest_result(result, index)
            row = {}
            for kind in ("gross", "net"):
                returns = getattr(sliced, f"{kind}_returns")
                low, high = sharpe_confidence_interval(returns)
                row[f"{kind}_sharpe"] = sharpe_ratio(returns)
                row[f"{kind}_ci_low"] = low
                row[f"{kind}_ci_high"] = high
            rows[(name, period)] = row
    table = pd.DataFrame(rows).T
    table.index.names = ["strategy", "period"]
    return table


def trial_register(data: study.StudyData) -> dict[str, BacktestResult]:
    """Every strategy configuration the project has backtested, once each:

    - momentum lookbacks from main.py's and optimize.py's grids, plus the
      1-day lookback the follow-up study added by flipping 1-day reversal;
    - every EMA pair from both grids;
    - optimize.py's reversal lookbacks;
    - the study's four 1-day variants (rank weights and three volume
      versions);
    - main.py's three ways of combining its selected signals.

    Left out on purpose: the study's one-day-delay run (a diagnostic, not
    something anyone would trade) and optimize.py's walk-forward strategies
    (they choose among these configurations rather than adding new ones).
    Work done outside these scripts isn't counted, so treat the total as a
    floor.
    """
    prices, returns = data.prices, data.returns
    signals = {}
    momentum_lookbacks = set(pipeline.MOMENTUM_LOOKBACK_CANDIDATES)
    momentum_lookbacks |= set(optimize.MOMENTUM_LOOKBACK_CANDIDATES) | {1}
    for lookback in sorted(momentum_lookbacks):
        signals[f"momentum_{lookback}d"] = time_horizon_momentum(prices, lookback=lookback)
    for fast, slow in sorted(set(pipeline.EMA_PAIR_CANDIDATES) | set(optimize.EMA_PAIR_CANDIDATES)):
        signals[f"ema_{fast}_{slow}"] = ema_crossover_momentum(
            prices, fast_span=fast, slow_span=slow
        )
    for lookback in optimize.REVERSAL_LOOKBACK_CANDIDATES:
        signals[f"reversal_{lookback}d"] = time_horizon_reversal(prices, lookback=lookback)
    volume = study.volume_signals(data)
    signals["momentum_1d_rank_weights"] = time_horizon_momentum(prices, lookback=1).rank(axis=1)
    signals["momentum_1d_high_volume"] = volume["momentum_1d, high-volume coins"]
    signals["momentum_1d_volume_weighted"] = volume["momentum_1d, volume-weighted"]
    signals["reversal_1d_low_volume"] = volume["reversal_1d, low-volume coins"]
    trials = run_multi_strategy_backtest(signals, returns)

    components, component_results, _, _ = _main_components(data)
    for method in ("equal", "inverse_vol", "ic"):
        combined_signal = combine_signals(
            components, method=method, **_combination_kwargs(method, component_results, returns)
        )
        trials[f"combined_{method}"] = UnconstrainedBacktester(combined_signal, returns).run()
    return trials


def _trial_returns(data: study.StudyData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gross and net daily returns of every trial, over the common window."""
    trials = trial_register(data)
    start = data.returns.index[COMMON_START_BAR]
    gross = pd.DataFrame({name: r.gross_returns for name, r in trials.items()}).loc[start:]
    net = pd.DataFrame({name: r.net_returns for name, r in trials.items()}).loc[start:]
    return gross, net


def trial_table(data: study.StudyData) -> pd.DataFrame:
    """Every trial's Sharpe over the common window, best before costs first."""
    gross, net = _trial_returns(data)
    table = pd.DataFrame(
        {"gross_sharpe": gross.apply(sharpe_ratio), "net_sharpe": net.apply(sharpe_ratio)}
    )
    table.index.name = "trial"
    return table.sort_values("gross_sharpe", ascending=False)


def deflation_table(data: study.StudyData) -> pd.DataFrame:
    """The best trial before costs and the best after costs, each with its
    interval, the probability its true Sharpe is above zero, and the
    deflated Sharpe ratio: the probability it beats the luck hurdle (the
    Sharpe the best of N no-edge strategies would show). N is counted two
    ways, as every trial and as the effective number of independent ones
    (from how closely the trials' gross returns move together)."""
    gross, net = _trial_returns(data)
    n_trials = gross.shape[1]
    n_independent = effective_number_of_trials(gross)

    rows = {}
    for label, frame in [("before costs", gross), ("after 20 bps costs", net)]:
        sharpes = frame.apply(sharpe_ratio)
        best = sharpes.idxmax()
        returns = frame[best]
        low, high = sharpe_confidence_interval(returns)
        rows[label] = {
            "best_trial": best,
            "sharpe": sharpes[best],
            "ci_low": low,
            "ci_high": high,
            "prob_above_zero": probabilistic_sharpe_ratio(returns),
            "trials": n_trials,
            "luck_hurdle_all": expected_maximum_sharpe(n_trials, len(returns)),
            "prob_beats_luck_all": deflated_sharpe_ratio(returns, n_trials),
            "independent_trials": n_independent,
            "luck_hurdle_independent": expected_maximum_sharpe(n_independent, len(returns)),
            "prob_beats_luck_independent": deflated_sharpe_ratio(returns, n_independent),
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
def format_intervals(table: pd.DataFrame) -> pd.DataFrame:
    """``interval_table`` with each interval as one "low to high" column."""
    formatted = pd.DataFrame(index=table.index)
    for kind in ("gross", "net"):
        formatted[f"{kind}_sharpe"] = table[f"{kind}_sharpe"].map("{:+.2f}".format)
        formatted[f"{kind}_95%_interval"] = [
            f"{low:+.2f} to {high:+.2f}"
            for low, high in zip(table[f"{kind}_ci_low"], table[f"{kind}_ci_high"], strict=True)
        ]
    return formatted


def format_deflation(table: pd.DataFrame) -> pd.DataFrame:
    """``deflation_table`` with probabilities as percentages and the
    interval as one column."""
    formatted = pd.DataFrame(index=table.index)
    formatted["best_trial"] = table["best_trial"]
    formatted["sharpe"] = table["sharpe"].map("{:+.2f}".format)
    formatted["95%_interval"] = [
        f"{low:+.2f} to {high:+.2f}"
        for low, high in zip(table["ci_low"], table["ci_high"], strict=True)
    ]
    formatted["P(true Sharpe > 0)"] = table["prob_above_zero"].map("{:.0%}".format)
    for suffix, count_column in [("all", "trials"), ("independent", "independent_trials")]:
        count = f"{table[count_column].iloc[0]:.0f}" if suffix == "all" else (
            f"{table[count_column].iloc[0]:.1f}"
        )
        formatted[f"luck hurdle, N={count}"] = table[f"luck_hurdle_{suffix}"].map("{:.2f}".format)
        formatted[f"P(beats luck), N={count}"] = table[f"prob_beats_luck_{suffix}"].map(
            "{:.0%}".format
        )
    return formatted


def main() -> None:
    data = study.load_study_data()

    print("=== 95% confidence intervals for every Sharpe in the README table ===")
    print(format_intervals(interval_table(data)).to_string())

    deflation = deflation_table(data)
    start = data.returns.index[COMMON_START_BAR].date()
    print(
        f"\n=== Deflated Sharpe ratio: the best of {deflation['trials'].iloc[0]:.0f} trials, "
        f"{start} -> {data.returns.index[-1].date()} ==="
    )
    print(format_deflation(deflation).T.to_string())

    print("\n=== Every trial, best before costs first ===")
    print(trial_table(data).map("{:+.2f}".format).to_string())

    logged_columns = ("best_trial", "sharpe", "prob_beats_luck_all", "prob_beats_luck_independent")
    log_run(
        "overfitting_check",
        parameters={"start": pipeline.START, "end": pipeline.END, "common_start": start},
        result_summary={
            "trials": deflation["trials"].iloc[0],
            "independent_trials": deflation["independent_trials"].iloc[0],
        }
        | {
            f"{label}_{column}": deflation.loc[label, column]
            for label in deflation.index
            for column in logged_columns
        },
    )
    print("\n(audit trail for this run: data/audit_log.jsonl)")


if __name__ == "__main__":
    main()
