"""Strategy combination / weighting — Feature 5 in specs/feature-list.md.

Blends several per-strategy signals (each a time x assets DataFrame, as
produced by signals/momentum.py, signals/reversal.py, or any other signal
function) into a single combined signal, using one of three weighting
methods (5.1-5.3), selected via ``combine_signals``'s ``method`` argument
(5.4). The combined signal is fed into ``backtest.UnconstrainedBacktester``
exactly like a single-strategy signal — this module only decides *how much*
weight each input strategy gets, not how the combined signal is traded.

Each weighting method needs different supporting data, so they're kept as
independent, separately testable functions rather than folded into one
do-everything call:

- ``equal_weights`` (5.1) needs only the strategy names.
- ``inverse_vol_weights`` (5.2) needs each strategy's own realized return
  series (e.g. ``BacktestResult.net_returns`` from
  ``backtest.run_multi_strategy_backtest``) — it down-weights noisier
  strategies.
- ``ic_weights`` (5.3) needs each strategy's raw signal plus forward asset
  returns — it up-weights strategies whose signal has historically ranked
  assets well (higher information coefficient).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd

WeightingMethod = Literal["equal", "inverse_vol", "ic"]


def standardize_signal(signal: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score of a signal, row by row.

    Individual strategy signals live on wildly different scales (e.g. raw
    pct-return momentum vs. a -1/0/1 discrete position), so they're put on a
    comparable footing before being weighted together. Rows with zero
    cross-sectional dispersion (or all-NaN) come back as all-NaN.
    """
    mean = signal.mean(axis=1)
    std = signal.std(axis=1)
    return signal.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)


# ---------------------------------------------------------------------------
# 5.1 Equal-weight combination
# ---------------------------------------------------------------------------
def equal_weights(names: Iterable[str]) -> pd.Series:
    """Equal weight across all named strategies."""
    names = list(names)
    if not names:
        raise ValueError("need at least one strategy to weight")
    return pd.Series(1.0 / len(names), index=names)


# ---------------------------------------------------------------------------
# 5.2 Inverse-volatility weighting
# ---------------------------------------------------------------------------
def inverse_vol_weights(strategy_returns: dict[str, pd.Series]) -> pd.Series:
    """Weight each strategy inversely proportional to its own return volatility.

    ``strategy_returns`` maps a strategy name to its realized (e.g.
    net-of-cost) return series — typically ``BacktestResult.net_returns``
    for each entry of ``backtest.run_multi_strategy_backtest``'s output.
    Noisier strategies get down-weighted relative to steadier ones.
    """
    if not strategy_returns:
        raise ValueError("need at least one strategy to weight")

    vols = pd.Series({name: returns.std() for name, returns in strategy_returns.items()})
    if (vols == 0).any() or vols.isna().any():
        raise ValueError("cannot inverse-vol weight a strategy with zero or undefined volatility")

    inverse_vol = 1.0 / vols
    return inverse_vol / inverse_vol.sum()


# ---------------------------------------------------------------------------
# 5.3 IC-weighted combination
# ---------------------------------------------------------------------------
def ic_weights(signals: dict[str, pd.DataFrame], forward_returns: pd.DataFrame) -> pd.Series:
    """Weight each strategy by its historical information coefficient (IC).

    For each strategy, the per-period IC is the cross-sectional rank
    correlation (Spearman) between that strategy's signal and the forward
    asset returns realized over the following period; the strategy's score
    is the average IC across all periods. Scores are floored at zero (a
    strategy with negative average IC contributes nothing rather than being
    combined with a flipped sign) and renormalized to sum to 1.

    Spearman correlation is computed by hand (rank each side, then Pearson-
    correlate the ranks) rather than via pandas' ``method="spearman"``, which
    requires ``scipy`` — not otherwise a project dependency.
    """
    if not signals:
        raise ValueError("need at least one strategy to weight")

    scores = {}
    for name, signal in signals.items():
        aligned_returns = forward_returns.reindex_like(signal)
        signal_ranks = signal.rank(axis=1)
        return_ranks = aligned_returns.rank(axis=1)
        per_period_ic = signal_ranks.corrwith(return_ranks, axis=1)
        scores[name] = per_period_ic.mean()

    ic_scores = pd.Series(scores).clip(lower=0.0)
    total = ic_scores.sum()
    if total == 0 or np.isnan(total):
        raise ValueError("no strategy has positive average IC; cannot IC-weight")
    return ic_scores / total


# ---------------------------------------------------------------------------
# 5.4 Configurable selection of weighting method
# ---------------------------------------------------------------------------
def combine_signals(
    signals: dict[str, pd.DataFrame],
    method: WeightingMethod = "equal",
    *,
    strategy_returns: dict[str, pd.Series] | None = None,
    forward_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine several strategy signals into one, weighted by ``method``.

    Each input signal is cross-sectionally standardized (``standardize_signal``)
    then summed using the weights ``method`` selects:

    - ``"equal"`` (5.1): no extra arguments needed.
    - ``"inverse_vol"`` (5.2): requires ``strategy_returns``.
    - ``"ic"`` (5.3): requires ``forward_returns``.

    The result is a single combined signal DataFrame, same shape as the
    inputs, ready to pass into ``backtest.UnconstrainedBacktester``.
    """
    if not signals:
        raise ValueError("need at least one strategy to combine")

    if method == "equal":
        weights = equal_weights(signals.keys())
    elif method == "inverse_vol":
        if strategy_returns is None:
            raise ValueError("'inverse_vol' weighting requires `strategy_returns`")
        weights = inverse_vol_weights(strategy_returns)
    elif method == "ic":
        if forward_returns is None:
            raise ValueError("'ic' weighting requires `forward_returns`")
        weights = ic_weights(signals, forward_returns)
    else:
        raise ValueError(f"Unknown method: {method!r} (expected 'equal', 'inverse_vol', or 'ic')")

    combined = None
    for name, signal in signals.items():
        weighted = standardize_signal(signal) * weights[name]
        combined = weighted if combined is None else combined.add(weighted, fill_value=0.0)
    return combined
