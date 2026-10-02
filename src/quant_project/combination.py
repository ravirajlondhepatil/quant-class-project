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

- ``equal_weights`` (5.1) needs only the strategy names. It's static (the
  same split at every bar), since there's no data-dependent estimate here
  to accidentally leak future information through.
- ``inverse_vol_weights`` (5.2) needs each strategy's own realized return
  series (e.g. ``BacktestResult.net_returns`` from
  ``backtest.run_multi_strategy_backtest``) — it down-weights noisier
  strategies.
- ``ic_weights`` (5.3) needs each strategy's raw signal plus forward asset
  returns — it up-weights strategies whose signal has historically ranked
  assets well (higher information coefficient).

``inverse_vol_weights`` and ``ic_weights`` return a *time-varying* weight
(one row per bar, not one static number per strategy) per
specs/methodology-fixes-scope.md B1: the weight used at bar t is estimated
only from data strictly before t (an expanding window, shifted so bar t's
own not-yet-fully-realized data never leaks in), not from the whole
backtest window including bars after t. ``ic_weights`` additionally
correlates each bar's signal against the return genuinely realized *after*
it, not the contemporaneous return at that bar (B2) — see that function's
docstring for exactly how.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd

WeightingMethod = Literal["equal", "inverse_vol", "ic"]

DEFAULT_MIN_PERIODS = 20
"""Minimum number of strictly-past observations required before a
time-varying weight is considered estimable, for both ``inverse_vol_weights``
and ``ic_weights``. ~20 bars (roughly a month of daily data) is a plain,
round warm-up length — not tuned to any dataset — chosen only so the
expanding-window estimate isn't drawn from a handful of points; any run
long enough for these weights to matter can afford a short warm-up where
the combined signal is simply not yet defined (see ``combine_signals``)."""


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
def inverse_vol_weights(
    strategy_returns: dict[str, pd.Series],
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> pd.DataFrame:
    """Time-varying weight, inversely proportional to each strategy's own
    trailing return volatility.

    ``strategy_returns`` maps a strategy name to its realized (e.g.
    net-of-cost) return series — typically ``BacktestResult.net_returns``
    for each entry of ``backtest.run_multi_strategy_backtest``'s output.

    Returns a DataFrame (time x strategy): the weight at row t is computed
    from each strategy's return volatility over an *expanding window of
    strictly-past bars* (``returns.shift(1)`` before the expanding
    calculation, so a weight set for bar t never reflects bar t's own
    return, let alone anything after it — B1). Before ``min_periods``
    strictly-past observations exist, or wherever a strategy's trailing
    volatility is exactly zero (which would otherwise divide by zero), that
    strategy's weight is NaN for that bar rather than raising — a
    multi-year backtest shouldn't abort over one quiet bar in one strategy.
    ``combine_signals`` naturally treats a NaN weight as "no opinion yet."
    """
    if not strategy_returns:
        raise ValueError("need at least one strategy to weight")

    returns_df = pd.DataFrame(strategy_returns)
    past_returns = returns_df.shift(1)
    expanding_vol = past_returns.expanding(min_periods=min_periods).std()
    expanding_vol = expanding_vol.where(expanding_vol > 0)

    inverse_vol = 1.0 / expanding_vol
    row_sum = inverse_vol.sum(axis=1)
    return inverse_vol.div(row_sum.where(row_sum > 0), axis=0)


# ---------------------------------------------------------------------------
# 5.3 IC-weighted combination
# ---------------------------------------------------------------------------
def ic_weights(
    signals: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> pd.DataFrame:
    """Time-varying weight, by each strategy's trailing information
    coefficient (IC).

    Returns a DataFrame (time x strategy). Two look-ahead fixes from
    specs/methodology-fixes-scope.md live here together, since they
    compound:

    - **B2** — the per-bar IC itself must correlate a signal against the
      return genuinely realized *after* it, not the contemporaneous return
      at the same bar. ``forward_returns`` is shifted by one bar
      (``forward_returns.shift(-1)``) before correlating, so the IC
      attributed to bar t uses ``forward_returns.loc[t + 1]`` — what a
      position opened at t would actually have earned — matching the lag
      convention ``backtest.UnconstrainedBacktester`` already uses.
    - **B1** — a weight set for bar t can only use IC values already
      knowable by t. The IC attributed to bar s isn't fully known until
      s + 1 (it needs that bar's forward return), so the weight at t
      averages IC values up to bar t - 1 only (``shift(1)`` then an
      expanding mean), never bar t's own IC or anything later.

    As with ``inverse_vol_weights``, a strategy's per-bar score is floored
    at zero (a strategy with non-positive trailing average IC contributes
    nothing rather than flipping sign into the combination) and
    renormalized across whichever strategies have a defined score that bar;
    before ``min_periods`` strictly-past, fully-realized IC observations
    exist, the weight is NaN.
    """
    if not signals:
        raise ValueError("need at least one strategy to weight")

    genuinely_forward_returns = forward_returns.shift(-1)

    per_period_ic = {}
    for name, signal in signals.items():
        aligned_returns = genuinely_forward_returns.reindex_like(signal)
        signal_ranks = signal.rank(axis=1)
        return_ranks = aligned_returns.rank(axis=1)
        per_period_ic[name] = signal_ranks.corrwith(return_ranks, axis=1)

    ic_df = pd.DataFrame(per_period_ic)
    expanding_ic = ic_df.shift(1).expanding(min_periods=min_periods).mean()
    scores = expanding_ic.clip(lower=0.0)

    row_sum = scores.sum(axis=1)
    return scores.div(row_sum.where(row_sum > 0), axis=0)


# ---------------------------------------------------------------------------
# 5.4 Configurable selection of weighting method
# ---------------------------------------------------------------------------
def combine_signals(
    signals: dict[str, pd.DataFrame],
    method: WeightingMethod = "equal",
    *,
    strategy_returns: dict[str, pd.Series] | None = None,
    forward_returns: pd.DataFrame | None = None,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> pd.DataFrame:
    """Combine several strategy signals into one, weighted by ``method``.

    Each input signal is cross-sectionally standardized (``standardize_signal``)
    then summed using the weights ``method`` selects:

    - ``"equal"`` (5.1): static — no extra arguments needed.
    - ``"inverse_vol"`` (5.2): time-varying — requires ``strategy_returns``.
    - ``"ic"`` (5.3): time-varying — requires ``forward_returns``.

    ``min_periods`` is forwarded to ``inverse_vol_weights``/``ic_weights``
    (ignored for ``"equal"``, which has nothing to estimate). Wherever a
    weight is undefined for *some but not all* strategies at a bar (the
    time-varying methods' warm-up period), the undefined ones contribute
    zero for that bar rather than NaN-ing out the whole combined row
    (``pandas``' ``fill_value=0.0`` only substitutes where the *other* side
    of an addition has a real value). If *every* strategy is undefined for
    a bar, there's no real value on either side, so the combined signal
    stays NaN there — which `backtest.build_dollar_neutral_weights`
    already treats identically to a flat-zero row: both come back as "no
    position".

    The result is a single combined signal DataFrame, same shape as the
    inputs, ready to pass into ``backtest.UnconstrainedBacktester``.
    """
    if not signals:
        raise ValueError("need at least one strategy to combine")

    any_signal = next(iter(signals.values()))

    if method == "equal":
        static_weights = equal_weights(signals.keys())
        weights = pd.DataFrame(
            {name: static_weights[name] for name in signals}, index=any_signal.index
        )
    elif method == "inverse_vol":
        if strategy_returns is None:
            raise ValueError("'inverse_vol' weighting requires `strategy_returns`")
        weights = inverse_vol_weights(strategy_returns, min_periods=min_periods)
    elif method == "ic":
        if forward_returns is None:
            raise ValueError("'ic' weighting requires `forward_returns`")
        weights = ic_weights(signals, forward_returns, min_periods=min_periods)
    else:
        raise ValueError(f"Unknown method: {method!r} (expected 'equal', 'inverse_vol', or 'ic')")

    combined = None
    for name, signal in signals.items():
        weighted = standardize_signal(signal).mul(weights[name], axis=0)
        combined = weighted if combined is None else combined.add(weighted, fill_value=0.0)
    return combined
