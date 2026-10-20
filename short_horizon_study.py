"""Follow-up study of the 1-day reversal result, from the mentor's second
project review (Oct 2026). ``main.py`` found 1-day reversal losing at a net
Sharpe of about -4 in both the training and the test period; the review
asked to (1) flip it and test the opposite trade, 1-day cross-sectional
momentum, (2) find out what drives a Sharpe that large before trusting it
-- costs and turnover, single coins, single periods -- and (3) try one
economically motivated extension, volume, rather than a wider parameter
search.

Each of those is one pre-specified experiment below. Nothing is tuned:
every parameter is fixed in advance with its reason stated where it is
defined, every variant is reported on *both* the training and the test
period (the same 70/30 split as ``main.py``), and nothing is chosen by its
test-period result. The notebook (``notebooks/performance_summary.ipynb``)
calls these same functions for its follow-up section, so the numbers it
discusses are the numbers this script prints.

Reading the tables: ``gross_sharpe`` is before costs, ``net_sharpe`` after
the course brief's 20 bps market-order cost, ``net_sharpe_limit`` after
its 7 bps limit-order cost. ``gross_t`` is the t-statistic of the mean
daily gross return (|t| > 2 is the usual bar for "distinguishable from
zero"). ``breakeven_bps`` is the cost per unit of turnover at which the
strategy would exactly break even (``costs.breakeven_cost_bps``) -- the
single number to compare against 7 and 20 bps.

Run: uv run python short_horizon_study.py
"""

from __future__ import annotations

from dataclasses import dataclass

import main as pipeline
import numpy as np
import pandas as pd

from quant_project.audit import log_run
from quant_project.backtest import (
    BacktestResult,
    UnconstrainedBacktester,
    run_multi_strategy_backtest,
)
from quant_project.costs import breakeven_cost_bps, net_of_costs
from quant_project.data import build_close_price_panel, build_field_panel
from quant_project.model_selection import TrainTestSplit, slice_backtest_result, split_prices
from quant_project.performance import DEFAULT_PERIODS_PER_YEAR, alpha_beta, sharpe_ratio
from quant_project.signals.momentum import (
    activity_weighted_momentum,
    informed_momentum,
    time_horizon_momentum,
)
from quant_project.signals.reversal import time_horizon_reversal, uninformed_reversal

ACTIVITY_LOOKBACK = 20
"""Relative volume = today's volume over its trailing 20-day mean. 20 days
(about a month) is the same plain warm-up length the combination weights
use (``combination.DEFAULT_MIN_PERIODS``) -- fixed, not tuned."""

VOLUME_QUANTILE = 0.3
"""High-volume coins are the top 30% by relative volume that day, low-volume
coins the bottom 30% (about 5 of 15 coins each). 0.3 is
``reversal.uninformed_reversal``'s existing default, mirrored for the
high-volume side -- fixed, not tuned."""

TREND_LOOKBACK = 30
"""Market regime for the up-/down-trend check: the sign of the
equal-weighted universe's trailing 30-day return, as known when the
position is set. A diagnostic split, not a traded parameter."""

N_BEST_DAYS = 5
"""How many of a period's best days the outlier check removes."""


@dataclass
class StudyData:
    prices: pd.DataFrame
    volume: pd.DataFrame
    returns: pd.DataFrame
    benchmark_returns: pd.Series
    split: TrainTestSplit

    @property
    def periods(self) -> dict[str, pd.Index]:
        return {"train": self.split.train_prices.index, "test": self.split.test_prices.index}


def load_study_data() -> StudyData:
    """``main.py``'s pinned sample and train/test split, plus volume."""
    universe = pipeline.load_universe()
    prices = build_close_price_panel(universe)
    volume = build_field_panel(universe, "volume").reindex_like(prices)
    returns = prices.pct_change()
    return StudyData(
        prices=prices,
        volume=volume,
        returns=returns,
        benchmark_returns=returns[pipeline.BENCHMARK_SYMBOL],
        split=split_prices(prices, train_fraction=pipeline.TRAIN_FRACTION),
    )


def _t_stat(returns: pd.Series) -> float:
    """t-statistic of the mean of a daily return series (null: mean is zero)."""
    returns = returns.dropna()
    std = returns.std()
    if not std > 0.0:
        return float("nan")
    return returns.mean() / std * np.sqrt(len(returns))


def period_stats(result: BacktestResult, index: pd.Index, benchmark_returns: pd.Series) -> dict:
    """Gross-vs-net summary of one backtest over one period (``index``)."""
    sliced = slice_backtest_result(result, index)
    gross, net, turnover = sliced.gross_returns, sliced.net_returns, sliced.turnover
    return {
        "gross_sharpe": sharpe_ratio(gross),
        "gross_t": _t_stat(gross),
        "net_sharpe": sharpe_ratio(net),
        "net_sharpe_limit": sharpe_ratio(net_of_costs(gross, turnover, order_type="limit")),
        "gross_ann_return": gross.mean() * DEFAULT_PERIODS_PER_YEAR,
        "cost_ann": (gross - net).mean() * DEFAULT_PERIODS_PER_YEAR,
        "turnover": turnover.mean(),
        "breakeven_bps": breakeven_cost_bps(gross, turnover),
        "net_alpha_t": alpha_beta(net, benchmark_returns).alpha_t_stat,
    }


def stats_table(results: dict[str, BacktestResult], data: StudyData) -> pd.DataFrame:
    """``period_stats`` for every strategy in ``results``, train and test."""
    rows = {
        (name, period): period_stats(result, index, data.benchmark_returns)
        for name, result in results.items()
        for period, index in data.periods.items()
    }
    table = pd.DataFrame(rows).T
    table.index.names = ["strategy", "period"]
    return table


# ---------------------------------------------------------------------------
# Experiment 1: flip the signal
# ---------------------------------------------------------------------------
def flip_experiment(data: StudyData) -> pd.DataFrame:
    """1-day reversal vs. its exact opposite, 1-day momentum.

    Flipping the signal flips every position, so it flips the *gross*
    return exactly -- but turnover, and therefore cost, is identical either
    way. Comparing gross with net here is the whole experiment.
    """
    signals = {
        "reversal_1d": time_horizon_reversal(data.prices, lookback=1),
        "momentum_1d": time_horizon_momentum(data.prices, lookback=1),
    }
    return stats_table(run_multi_strategy_backtest(signals, data.returns), data)


# ---------------------------------------------------------------------------
# Experiment 2: what drives the result
# ---------------------------------------------------------------------------
def _momentum_1d(data: StudyData) -> pd.DataFrame:
    return time_horizon_momentum(data.prices, lookback=1)


def by_year(data: StudyData) -> pd.DataFrame:
    """1-day momentum, one row per calendar year (train/test boundary falls
    inside 2025 -- see ``data.split``)."""
    result = UnconstrainedBacktester(_momentum_1d(data), data.returns).run()
    rows = {}
    for year, dates in data.returns.groupby(data.returns.index.year).groups.items():
        sliced = slice_backtest_result(result, pd.DatetimeIndex(dates))
        rows[year] = {
            "days": len(dates),
            "gross_sharpe": sharpe_ratio(sliced.gross_returns),
            "gross_t": _t_stat(sliced.gross_returns),
            "net_sharpe": sharpe_ratio(sliced.net_returns),
            "breakeven_bps": breakeven_cost_bps(sliced.gross_returns, sliced.turnover),
        }
    table = pd.DataFrame(rows).T
    table.index.name = "year"
    return table


def by_coin(data: StudyData) -> pd.DataFrame:
    """Each coin's share of 1-day momentum's gross return (annualized, so the
    rows sum to the strategy's gross annual return), and the gross Sharpe
    left over when that coin is removed from the universe entirely."""
    signal = _momentum_1d(data)
    result = UnconstrainedBacktester(signal, data.returns).run()
    daily_pnl = result.weights * data.returns

    columns = {}
    for period, index in data.periods.items():
        columns[(period, "gross_contribution_ann")] = (
            daily_pnl.reindex(index).mean() * DEFAULT_PERIODS_PER_YEAR
        )
        leave_one_out = {}
        for coin in data.prices.columns:
            others = data.prices.columns.drop(coin)
            dropped = UnconstrainedBacktester(signal[others], data.returns[others]).run()
            leave_one_out[coin] = sharpe_ratio(dropped.gross_returns.reindex(index))
        columns[(period, "gross_sharpe_without_coin")] = pd.Series(leave_one_out)

    table = pd.DataFrame(columns)
    table.index.name = "coin"
    return table.sort_values(("test", "gross_contribution_ann"), ascending=False)


def robustness_checks(data: StudyData) -> pd.DataFrame:
    """Does 1-day momentum's gross result survive changes that *shouldn't*
    matter if the effect is real?

    - rank weights: weight by cross-sectional rank instead of raw return
      size, so one coin's outsized move can't dominate a day's book;
    - one-day execution delay: trade a day later than the signal allows --
      a real effect should fade gradually, not vanish;
    - best days removed: drop the period's ``N_BEST_DAYS`` best days;
    - market regime: up-trend vs. down-trend days, testing the notebook's
      earlier "it's the bull market" explanation directly.
    """
    base_signal = _momentum_1d(data)
    variants = run_multi_strategy_backtest(
        {
            "baseline": base_signal,
            "rank weights": base_signal.rank(axis=1),
            "one-day execution delay": base_signal.shift(1),
        },
        data.returns,
    )
    base_gross = variants["baseline"].gross_returns

    market = data.returns.mean(axis=1)
    trailing_trend = (1 + market).rolling(TREND_LOOKBACK).apply(np.prod, raw=True) - 1
    # The position held on day t was set at the close of t-1, so classify
    # day t by the trend as known at t-1.
    up_trend = trailing_trend.shift(1) > 0
    down_trend = trailing_trend.shift(1) < 0

    rows = {}
    for period, index in data.periods.items():
        for name, result in variants.items():
            gross = result.gross_returns.reindex(index)
            rows[(name, period)] = {
                "days": len(gross),
                "gross_sharpe": sharpe_ratio(gross),
                "gross_t": _t_stat(gross),
            }
        gross = base_gross.reindex(index)
        for n_days, label in [(1, "best day"), (N_BEST_DAYS, f"{N_BEST_DAYS} best days")]:
            trimmed = gross.drop(gross.nlargest(n_days).index)
            rows[(f"baseline minus {label}", period)] = {
                "days": len(trimmed),
                "gross_sharpe": sharpe_ratio(trimmed),
                "gross_t": _t_stat(trimmed),
            }
        for regime, mask in [("up-trend days", up_trend), ("down-trend days", down_trend)]:
            subset = gross[mask.reindex(index, fill_value=False)]
            rows[(f"baseline, {regime}", period)] = {
                "days": len(subset),
                "gross_sharpe": sharpe_ratio(subset),
                "gross_t": _t_stat(subset),
            }

    table = pd.DataFrame(rows).T
    table.index.names = ["check", "period"]
    return table


def best_days(data: StudyData) -> pd.DataFrame:
    """1-day momentum's ``N_BEST_DAYS`` best days in each period, with the
    coin that contributed most that day: what the robustness check's
    "minus best days" rows are removing. ``share_of_period_gross`` is that
    day's share of the period's total gross return -- NaN when the period
    lost money overall, since a share of a negative total means nothing."""
    result = UnconstrainedBacktester(_momentum_1d(data), data.returns).run()
    daily_pnl = result.weights * data.returns

    rows = {}
    for period, index in data.periods.items():
        gross = result.gross_returns.reindex(index)
        total = gross.sum()
        for day in gross.nlargest(N_BEST_DAYS).index:
            top_coin = daily_pnl.loc[day].idxmax()
            rows[(period, day.date())] = {
                "gross_return": gross[day],
                "share_of_period_gross": gross[day] / total if total > 0 else float("nan"),
                "sigmas": gross[day] / gross.std(),
                "top_coin": top_coin,
                "top_coin_weight": result.weights.loc[day, top_coin],
                "top_coin_return": data.returns.loc[day, top_coin],
            }
    table = pd.DataFrame(rows).T
    table.index.names = ["period", "date"]
    return table


# ---------------------------------------------------------------------------
# Experiment 3: volume
# ---------------------------------------------------------------------------
def volume_signals(data: StudyData) -> dict[str, pd.DataFrame]:
    prices, volume = data.prices, data.volume
    return {
        "momentum_1d, all coins": time_horizon_momentum(prices, lookback=1),
        "momentum_1d, high-volume coins": informed_momentum(
            prices, volume, 1, ACTIVITY_LOOKBACK, high_activity_quantile=1 - VOLUME_QUANTILE
        ),
        "momentum_1d, volume-weighted": activity_weighted_momentum(
            prices, volume, return_lookback=1, activity_lookback=ACTIVITY_LOOKBACK
        ),
        "reversal_1d, all coins": time_horizon_reversal(prices, lookback=1),
        "reversal_1d, low-volume coins": uninformed_reversal(
            prices, volume, 1, ACTIVITY_LOOKBACK, low_activity_quantile=VOLUME_QUANTILE
        ),
    }


def volume_experiment(data: StudyData) -> pd.DataFrame:
    """Hypotheses (Llorente, Michaely, Saar & Wang, 2002): a move made on
    heavy volume is more likely informed and should continue; a move made on
    light volume is more likely noise or liquidity-driven and should
    reverse. Each conditioned strategy is compared with its unconditioned
    version on the same days."""
    results = run_multi_strategy_backtest(volume_signals(data), data.returns)
    return stats_table(results, data)


def volume_spread_test(data: StudyData) -> pd.DataFrame:
    """The two volume hypotheses as one test: the daily gross return of
    1-day momentum among high-volume coins minus 1-day momentum among
    low-volume coins. If heavy-volume moves continue and light-volume moves
    reverse, this spread is positive."""
    signals = volume_signals(data)
    results = run_multi_strategy_backtest(
        {
            "high": signals["momentum_1d, high-volume coins"],
            # Momentum among the low-volume coins is exactly the negative of
            # reversal among them.
            "low": -signals["reversal_1d, low-volume coins"],
        },
        data.returns,
    )
    spread = results["high"].gross_returns - results["low"].gross_returns

    rows = {}
    for period, index in data.periods.items():
        period_spread = spread.reindex(index)
        rows[period] = {
            "spread_ann": period_spread.mean() * DEFAULT_PERIODS_PER_YEAR,
            "spread_t": _t_stat(period_spread),
        }
    table = pd.DataFrame(rows).T
    table.index.name = "period"
    return table


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
_PERCENT_COLUMNS = {
    "gross_ann_return",
    "cost_ann",
    "gross_contribution_ann",
    "spread_ann",
    "gross_return",
    "share_of_period_gross",
    "top_coin_return",
}


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    """String-formatted copy of a study table for printing: percentages for
    return columns, two decimals for everything else, day counts as ints,
    text left as is, and "n/a" for missing values."""

    def _formatter(template: str):
        return lambda value: "n/a" if pd.isna(value) else template.format(value)

    formatted = table.copy().astype(object)
    for column in table.columns:
        name = column[-1] if isinstance(column, tuple) else column
        if name in _PERCENT_COLUMNS:
            formatted[column] = table[column].map(_formatter("{:+.1%}"))
        elif name == "days":
            formatted[column] = table[column].map(_formatter("{:.0f}"))
        elif name != "top_coin":
            formatted[column] = table[column].map(_formatter("{:+.2f}"))
    return formatted


def _print_section(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(format_table(table).to_string())


def main() -> None:
    data = load_study_data()
    print(
        f"Sample: {data.prices.index[0].date()} -> {data.prices.index[-1].date()}, "
        f"{data.prices.shape[1]} coins; train ends {data.split.train_prices.index[-1].date()}, "
        f"test starts {data.split.split_date.date()}"
    )

    flip = flip_experiment(data)
    _print_section("1. Flip the signal: 1-day reversal vs. 1-day momentum", flip)
    _print_section("2a. 1-day momentum by calendar year", by_year(data))
    _print_section("2b. 1-day momentum by coin", by_coin(data))
    _print_section("2c. 1-day momentum robustness checks (gross)", robustness_checks(data))
    _print_section("2d. 1-day momentum's best days", best_days(data))
    volume = volume_experiment(data)
    _print_section("3a. Volume-conditioned 1-day signals", volume)
    spread = volume_spread_test(data)
    _print_section("3b. High-volume minus low-volume momentum (gross spread)", spread)

    log_run(
        "short_horizon_study",
        parameters={
            "start": pipeline.START,
            "end": pipeline.END,
            "train_fraction": pipeline.TRAIN_FRACTION,
            "activity_lookback": ACTIVITY_LOOKBACK,
            "volume_quantile": VOLUME_QUANTILE,
        },
        result_summary={
            f"{name}_{period}_{metric}": flip.loc[(name, period), metric]
            for name in ("reversal_1d", "momentum_1d")
            for period in ("train", "test")
            for metric in ("gross_sharpe", "net_sharpe")
        }
        | {f"volume_spread_{period}_t": spread.loc[period, "spread_t"] for period in spread.index},
    )
    print("\n(audit trail for this run: data/audit_log.jsonl)")


if __name__ == "__main__":
    main()
