"""Unconstrained backtest engine — Feature 4 (4.1-4.4, 4.6) in
specs/feature-list.md.

"Unconstrained" is the WSQ course's own term (its "PythonUnconBacktest"
lecture) and matches standard quant-research usage: portfolio weights are
derived directly (proportionally) from the signal, long *and* short, with
no position limits, leverage cap, or sector/exposure constraints — as
opposed to a "constrained" backtest (e.g. long-only, capped position sizes).

Execution costs (4.5) live in costs.py, kept separate the way the course
separates its "Tcosts"/"Turnover" material from the backtest engine itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_project.costs import OrderType, net_of_costs


def build_dollar_neutral_weights(signal: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw signal into dollar-neutral portfolio weights.

    Each row is demeaned (so the cross-sectional average weight is zero,
    i.e. equal long and short dollar exposure) and scaled so gross exposure
    (sum of absolute weights) is 1.0. Rows with zero cross-sectional
    dispersion (or all-NaN signal) come back as all-NaN — callers should
    treat NaN weights as "no position".
    """
    demeaned = signal.sub(signal.mean(axis=1), axis=0)
    gross = demeaned.abs().sum(axis=1)
    weights = demeaned.div(gross.replace(0.0, np.nan), axis=0)
    return weights.astype(float)


@dataclass
class BacktestResult:
    weights: pd.DataFrame
    """Positions actually held each period (already lagged vs. the signal)."""
    turnover: pd.Series
    """Fraction of gross exposure traded each period."""
    gross_returns: pd.Series
    """Portfolio returns before execution costs."""
    net_returns: pd.Series
    """Portfolio returns after execution costs."""


class UnconstrainedBacktester:
    """Signal -> positions -> turnover -> net-of-cost returns.

    Parameters
    ----------
    signal: cross-sectional signal DataFrame (time x assets), higher = more
        attractive to be long.
    returns: forward per-period asset returns aligned to ``signal.index``
        (i.e. ``returns.loc[t]`` is the return earned from holding over the
        period starting at ``t``).
    order_type: "market" or "limit" — passed to the execution cost model.
    rebalance_every: hold the signal-implied weight fixed for this many
        periods instead of re-weighting every bar (reduces turnover/costs
        at the expense of reacting more slowly to the signal). Default 1
        rebalances every bar.
    start, end: optional date-range bounds (anything ``.loc[start:end]``
        accepts) — the backtest runs only over this slice of the inputs.
    """

    def __init__(
        self,
        signal: pd.DataFrame,
        returns: pd.DataFrame,
        order_type: OrderType = "market",
        rebalance_every: int = 1,
        start=None,
        end=None,
    ):
        if rebalance_every < 1:
            raise ValueError("rebalance_every must be >= 1")
        self.signal = signal.loc[start:end]
        self.returns = returns.loc[start:end]
        self.order_type = order_type
        self.rebalance_every = rebalance_every

    def run(self) -> BacktestResult:
        raw_weights = build_dollar_neutral_weights(self.signal)

        if self.rebalance_every > 1:
            # Keep only every Nth row as a "fresh" rebalance, hold it
            # constant in between by forward-filling.
            is_rebalance_bar = pd.Series(
                np.arange(len(raw_weights)) % self.rebalance_every == 0,
                index=raw_weights.index,
            )
            raw_weights = raw_weights.where(is_rebalance_bar).ffill()

        # Lag by one period: a signal/weight observed at the close of t is
        # traded into a position that earns the return realized over t -> t+1.
        positions = raw_weights.shift(1).reindex(self.returns.index).fillna(0.0)

        gross_returns = (positions * self.returns).sum(axis=1)

        # Turnover = change in target weight period over period (the first
        # bar's turnover is the cost of putting the initial position on).
        # This ignores weight drift from price moves between rebalances —
        # a standard simplification for a research-stage backtest.
        turnover = positions.diff().abs().sum(axis=1)
        turnover = turnover.fillna(positions.abs().sum(axis=1))

        net_returns = net_of_costs(gross_returns, turnover, order_type=self.order_type)

        return BacktestResult(
            weights=positions,
            turnover=turnover,
            gross_returns=gross_returns,
            net_returns=net_returns,
        )


def run_multi_strategy_backtest(
    signals: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    order_type: OrderType = "market",
    rebalance_every: int = 1,
    start=None,
    end=None,
) -> dict[str, BacktestResult]:
    """Run the same unconstrained backtest independently for several named
    strategies (Feature 4.3 — multiple candidate strategies in one run).

    Each strategy is backtested on its own for side-by-side comparison
    (e.g. time-horizon momentum vs. EMA-crossover momentum). Blending them
    into a single combined strategy is Feature 5 (Strategy Combination /
    Weighting), not this function.
    """
    return {
        name: UnconstrainedBacktester(
            signal,
            returns,
            order_type=order_type,
            rebalance_every=rebalance_every,
            start=start,
            end=end,
        ).run()
        for name, signal in signals.items()
    }
