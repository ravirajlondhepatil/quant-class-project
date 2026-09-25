"""Reversal signal construction — Feature 3 in specs/feature-list.md.

Every function here answers the opposite question from momentum.py: "has
this asset moved too far, too fast, and is likely to bounce back?" They
fall into the same two families as momentum.py:

- **Cross-sectional score** functions (``time_horizon_reversal``,
  ``uninformed_reversal``, ``correlation_reversal``,
  ``macro_conditioned_reversal``) return a continuous DataFrame (time x
  assets), higher = more attractive to be long (i.e. more "oversold").
  These map 1:1 onto the WSQ course's own "Reversal" module structure
  (Time Horizon / Uninformed Trading / Correlation / Macro), which is also
  what ref/ClassProject.docx's "How to find reversal" section summarizes —
  specs/feature-list.md 3.1-3.4.

- **Discrete position** function (``pairs_trading_distance_signal``)
  returns -1/0/+1 per asset per bar, matching a published technique (3.5):
  the "distance method" from Gatev, Goetzmann & Rouwenhorst (2006),
  "Pairs Trading: Performance of a Relative-Value Arbitrage Rule" — open a
  pair position when the price spread diverges beyond N historical standard
  deviations, close on convergence.

Feed either family into ``backtest.UnconstrainedBacktester``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 3.1 Time-horizon reversal
# ---------------------------------------------------------------------------
def time_horizon_reversal(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Negative of past return over ``lookback`` periods.

    Hypothesis: shorter lookbacks tend to show reversal (the classic
    short-term reversal factor). Test several ``lookback`` values (e.g. 1,
    2, 3, 5 bars) to map out where reversal is strongest.
    """
    return -prices.pct_change(lookback)


# ---------------------------------------------------------------------------
# 3.2 Uninformed-trading-conditioned reversal
# ---------------------------------------------------------------------------
def uninformed_reversal(
    prices: pd.DataFrame,
    activity: pd.DataFrame,
    lookback: int,
    activity_lookback: int,
    low_activity_quantile: float = 0.3,
) -> pd.DataFrame:
    """Reversal isolated to periods of low ("uninformed"/liquidity-driven)
    trading activity.

    Hypothesis: uninformed, liquidity-driven moves reverse more than
    information-driven moves (Campbell, Grossman & Wang, 1993; the course
    notes' "Fire Sale" crypto liquidation case is a specific instance).
    ``activity`` is any proxy relative to its own trailing average (volume
    is the default, same convention as ``momentum.activity_weighted_momentum``);
    the reversal signal is zeroed out (as NaN) wherever relative activity is
    *above* the ``low_activity_quantile`` cross-sectional cutoff for that bar.
    """
    reversal = time_horizon_reversal(prices, lookback)
    avg_activity = activity.rolling(activity_lookback).mean()
    relative_activity = activity / avg_activity

    cutoff = relative_activity.quantile(low_activity_quantile, axis=1)
    is_low_activity = relative_activity.le(cutoff, axis=0)
    return reversal.where(is_low_activity)


# ---------------------------------------------------------------------------
# 3.3 Correlation / pairs reversal (continuous z-score)
# ---------------------------------------------------------------------------
def correlation_reversal(
    prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    lookback: int,
    zscore_window: int,
) -> pd.DataFrame:
    """Mean-reversion signal on the return spread of correlated pairs/baskets.

    ``pairs`` is a list of ``(asset_a, asset_b)`` column-name tuples assumed
    to be correlated. For each pair, the signal is the negative z-score of
    the ``lookback``-period return spread over ``zscore_window`` bars: a
    stretched-wide spread (A up a lot relative to B) signals A should
    underperform / B should outperform going forward.

    Returns a DataFrame with the same shape as ``prices``; assets not in any
    pair are left as NaN.
    """
    returns = prices.pct_change(lookback)
    signal = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for asset_a, asset_b in pairs:
        spread = returns[asset_a] - returns[asset_b]
        rolling_spread = spread.rolling(zscore_window)
        spread_z = (spread - rolling_spread.mean()) / rolling_spread.std()
        signal[asset_a] = -spread_z
        signal[asset_b] = spread_z

    return signal


# ---------------------------------------------------------------------------
# 3.4 Macro-regime-conditioned reversal
# ---------------------------------------------------------------------------
def macro_conditioned_reversal(
    prices: pd.DataFrame,
    lookback: int,
    dislocation_indicator: pd.Series,
    dislocation_threshold_quantile: float = 0.7,
) -> pd.DataFrame:
    """Reversal scaled up during high-volatility / dislocated macro regimes.

    ``dislocation_indicator`` is a single time series aligned to
    ``prices.index`` (e.g. realized volatility, implied volatility, return
    dispersion, or average pairwise correlation — the indicators
    ref/ClassProject.docx names). The base reversal signal is multiplied by
    the indicator's rolling-rank (0-1) so reversal is stronger when the
    indicator is elevated relative to its own history, and floored to zero
    below ``dislocation_threshold_quantile`` so the signal is only active in
    sufficiently dislocated regimes.
    """
    reversal = time_horizon_reversal(prices, lookback)
    indicator = dislocation_indicator.reindex(prices.index)
    percentile_rank = indicator.rank(pct=True)

    active = percentile_rank >= dislocation_threshold_quantile
    scale = percentile_rank.where(active, 0.0)
    return reversal.mul(scale, axis=0)


# ---------------------------------------------------------------------------
# 3.5 Pairs trading, distance method — Gatev, Goetzmann & Rouwenhorst (2006)
# ---------------------------------------------------------------------------
def _single_pair_distance_positions(
    price_a: pd.Series,
    price_b: pd.Series,
    formation_window: int,
    entry_z: float,
    exit_z: float,
) -> tuple[pd.Series, pd.Series]:
    """Stateful open/close position path for one pair. See
    ``pairs_trading_distance_signal`` for the rule definitions."""
    norm_a = price_a / price_a.shift(formation_window)
    norm_b = price_b / price_b.shift(formation_window)
    spread = norm_a - norm_b
    spread_std = spread.rolling(formation_window).std()

    spread_vals = spread.to_numpy()
    std_vals = spread_std.to_numpy()

    positions_a = np.zeros(len(price_a), dtype=int)
    current = 0
    for t in range(len(price_a)):
        if np.isnan(std_vals[t]):
            positions_a[t] = current
            continue
        if current == 0:
            if spread_vals[t] >= entry_z * std_vals[t]:
                current = -1  # A outperformed B -> short A / long B
            elif spread_vals[t] <= -entry_z * std_vals[t]:
                current = 1  # A underperformed B -> long A / short B
        elif abs(spread_vals[t]) <= exit_z * std_vals[t]:
            current = 0
        positions_a[t] = current

    positions_a_series = pd.Series(positions_a, index=price_a.index)
    return positions_a_series, -positions_a_series


def pairs_trading_distance_signal(
    prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    formation_window: int,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
) -> pd.DataFrame:
    """Distance-method pairs trading (Gatev, Goetzmann & Rouwenhorst, 2006).

    For each pair, normalize both prices to a cumulative-return index over
    the trailing ``formation_window`` (``price / price.shift(formation_window)``)
    and take the spread between them. Open a position the bar the spread
    diverges beyond ``entry_z`` historical standard deviations (long the
    underperformer / short the outperformer); close it once the spread
    converges back within ``exit_z`` standard deviations.

    This uses a *rolling* formation window rather than the original paper's
    fixed 12-month-formation / 6-month-trading cycle, so it can run as a
    single continuous signal; the entry/exit rule itself matches the paper.

    Parameters
    ----------
    prices: DataFrame of prices (time x assets).
    pairs: list of ``(asset_a, asset_b)`` column-name tuples.
    formation_window: bars used both to build the normalized price index and
        to estimate the spread's historical standard deviation.
    entry_z: standard-deviation divergence that opens a position (paper: 2.0).
    exit_z: standard-deviation convergence that closes a position (0.0 =
        wait for the spread to cross back through zero).

    Returns
    -------
    DataFrame of positions in {-1, 0, 1}, same shape as ``prices``; assets
    not in any pair are left as NaN.
    """
    signal = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for asset_a, asset_b in pairs:
        positions_a, positions_b = _single_pair_distance_positions(
            prices[asset_a], prices[asset_b], formation_window, entry_z, exit_z
        )
        signal[asset_a] = positions_a
        signal[asset_b] = positions_b
    return signal
