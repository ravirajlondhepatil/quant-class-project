"""Momentum signal construction — Feature 2 in specs/feature-list.md.

Every function here answers the same question in a different way: "is this
asset likely to keep moving the way it's been moving?" They fall into two
families:

- **Cross-sectional score** functions (``time_horizon_momentum``,
  ``activity_weighted_momentum``, ``seasonality_momentum``, ``theme_momentum``,
  ``flow_window_momentum``) return a continuous DataFrame (time x assets)
  that is *higher* for assets expected to keep outperforming. These are our
  own formulations, from specs/feature-list.md 2.1-2.5.

- **Discrete position** functions (``channel_breakout_momentum``,
  ``ema_crossover_momentum``) return -1/0/+1 per asset per bar, matching two
  published/practitioner techniques transcribed from the course notes
  (2.6-2.7): a Donchian-style channel breakout, and the fast/slow EMA
  crossover from Rohrbach et al., "Momentum and Trend Following Trading
  Strategies for Currencies Revisited".

Feed either family into ``backtest.UnconstrainedBacktester`` once that
exists (Phase 3) — for now these are standalone, testable signal functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 2.1 Time-horizon momentum
# ---------------------------------------------------------------------------
def time_horizon_momentum(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Past return over ``lookback`` periods.

    Hypothesis: longer lookbacks tend to show momentum. Test several
    ``lookback`` values (e.g. 1, 3, 7, 14, 30, 90 bars) to map out where
    momentum appears vs. reverses.
    """
    return prices.pct_change(lookback)


# ---------------------------------------------------------------------------
# 2.2 Activity-weighted momentum
# ---------------------------------------------------------------------------
def activity_weighted_momentum(
    prices: pd.DataFrame,
    activity: pd.DataFrame,
    return_lookback: int,
    activity_lookback: int,
) -> pd.DataFrame:
    """Momentum scaled by relative activity / new information.

    Hypothesis: momentum is stronger when heightened activity coincides with
    the price move. ``activity`` is any proxy relative to its own trailing
    average — trading volume is the default proxy; a news/social/Google
    Trends intensity score (see the course notes' "Event-Driven" strategy)
    would plug into this same function unchanged.
    """
    raw_momentum = time_horizon_momentum(prices, return_lookback)
    avg_activity = activity.rolling(activity_lookback).mean()
    relative_activity = activity / avg_activity
    return raw_momentum * relative_activity


# ---------------------------------------------------------------------------
# 2.3 Seasonality-conditioned momentum
# ---------------------------------------------------------------------------
def seasonality_momentum(
    prices: pd.DataFrame,
    session_labels: pd.Series,
    lookback_sessions: int = 4,
) -> pd.DataFrame:
    """Momentum conditioned on recurring "seasons" (e.g. weekday vs. weekend,
    day vs. night, institutional vs. retail hours).

    ``session_labels`` is a Series aligned to ``prices.index`` giving a
    category label per bar. For each bar, the signal is the asset's average
    return over the previous ``lookback_sessions`` occurrences of the *same*
    session label — i.e. "how has this asset tended to do in sessions like
    this one".
    """
    returns = prices.pct_change()
    aligned_labels = session_labels.reindex(prices.index)

    signal = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for label in aligned_labels.dropna().unique():
        mask = aligned_labels == label
        same_session_returns = returns[mask]
        avg_same_session = same_session_returns.rolling(lookback_sessions, min_periods=1).mean()
        signal.loc[mask] = avg_same_session.reindex(signal.index[mask])
    return signal


# ---------------------------------------------------------------------------
# 2.4 Investment-theme basket momentum
# ---------------------------------------------------------------------------
def theme_momentum(prices: pd.DataFrame, theme_map: dict[str, str], lookback: int) -> pd.DataFrame:
    """Cross-sectional momentum of an asset's investment "theme" basket.

    ``theme_map`` maps each column in ``prices`` to a theme name (e.g.
    ``{"SOL": "L1", "AVAX": "L1", "UNI": "DeFi", ...}``). The signal for each
    asset is the past ``lookback``-period return of the equal-weighted
    basket of *other* assets sharing its theme — i.e. does the theme have
    momentum, independent of the single asset's own idiosyncratic move.
    Assets with no theme peers get NaN.
    """
    returns = prices.pct_change(lookback)
    themes = pd.Series(theme_map)
    signal = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for asset in prices.columns:
        theme = themes.get(asset)
        peers = [a for a in prices.columns if themes.get(a) == theme and a != asset]
        if not peers:
            signal[asset] = float("nan")
            continue
        signal[asset] = returns[peers].mean(axis=1)
    return signal


# ---------------------------------------------------------------------------
# 2.5 Flow-window momentum
# ---------------------------------------------------------------------------
def flow_window_momentum(
    prices: pd.DataFrame,
    window_mask: pd.Series,
    lookback_occurrences: int = 4,
) -> pd.DataFrame:
    """Momentum around a recurring, predictable flow window.

    ``window_mask`` is a boolean Series aligned to ``prices.index`` that is
    True during the recurring window of interest (e.g. institutional
    trading hours, or a scheduled rebalance window). The signal is the
    return realized during the most recent ``lookback_occurrences``
    instances of that window, intended to test whether the same mechanical
    flow is likely to repeat (i.e. can be front-run).
    """
    returns = prices.pct_change()
    window_mask = window_mask.reindex(prices.index).fillna(False)

    # Filter to only window bars first, then roll *within that subsequence* —
    # rolling on the full index would average over the last N calendar bars,
    # not the last N occurrences of the window (wrong whenever the window is
    # sparser than the calendar, e.g. a once-a-day window on hourly data).
    window_returns = returns[window_mask]
    avg_window_return = window_returns.rolling(lookback_occurrences, min_periods=1).mean()

    signal = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    signal.loc[window_mask] = avg_window_return.reindex(signal.index[window_mask])
    return signal


# ---------------------------------------------------------------------------
# 2.6 Channel breakout (Donchian-style) — matches course notes Strategy A
# ---------------------------------------------------------------------------
def _single_asset_channel_breakout(
    high: pd.Series,
    low: pd.Series,
    entry_lookback: int,
    exit_lookback_long: int,
    exit_lookback_short: int,
) -> pd.Series:
    """Stateful long/flat/short position path for one asset. See
    ``channel_breakout_momentum`` for the rule definitions."""
    entry_high = high.rolling(entry_lookback).max().shift(1)
    entry_low = low.rolling(entry_lookback).min().shift(1)
    exit_low = low.rolling(exit_lookback_long).min().shift(1)
    exit_high = high.rolling(exit_lookback_short).max().shift(1)

    high_vals = high.to_numpy()
    low_vals = low.to_numpy()
    entry_high_vals = entry_high.to_numpy()
    entry_low_vals = entry_low.to_numpy()
    exit_low_vals = exit_low.to_numpy()
    exit_high_vals = exit_high.to_numpy()

    positions = np.zeros(len(high), dtype=int)
    current = 0
    for t in range(len(high)):
        if current == 0:
            if not np.isnan(entry_high_vals[t]) and high_vals[t] >= entry_high_vals[t]:
                current = 1
            elif not np.isnan(entry_low_vals[t]) and low_vals[t] <= entry_low_vals[t]:
                current = -1
        elif (
            current == 1 and not np.isnan(exit_low_vals[t]) and low_vals[t] <= exit_low_vals[t]
        ) or (
            current == -1 and not np.isnan(exit_high_vals[t]) and high_vals[t] >= exit_high_vals[t]
        ):
            current = 0
        positions[t] = current

    return pd.Series(positions, index=high.index)


def channel_breakout_momentum(
    high: pd.DataFrame,
    low: pd.DataFrame,
    entry_lookback: int,
    exit_lookback_long: int,
    exit_lookback_short: int,
) -> pd.DataFrame:
    """Donchian-style channel breakout: trend confirmation via new highs/lows.

    Rules (per asset, per bar t):

    - Long entry:  ``high[t] >= max(high[t-N : t-1])`` -> go long
    - Long exit:   ``low[t]  <= min(low[t-X : t-1])``   -> flatten
    - Short entry: ``low[t]  <= min(low[t-N : t-1])``   -> go short
    - Short exit:  ``high[t] >= max(high[t-M : t-1])``  -> flatten

    where ``N = entry_lookback``, ``X = exit_lookback_long``,
    ``M = exit_lookback_short``. Exit windows are conventionally tighter than
    the entry window (X, M < N) so a position is cut faster than it was
    opened; this isn't enforced here since it's a research parameter choice.

    Unlike the other momentum functions, this is path-dependent (a position
    persists until its exit condition fires), so it's computed per asset
    with an explicit state machine rather than a single vectorized op.

    Parameters
    ----------
    high, low: DataFrames of daily high/low prices (time x assets), same
        shape and index.

    Returns
    -------
    DataFrame of positions in {-1, 0, 1}, same shape as ``high``.
    """
    if list(high.columns) != list(low.columns):
        raise ValueError("high and low must have the same columns in the same order")

    return pd.DataFrame(
        {
            asset: _single_asset_channel_breakout(
                high[asset], low[asset], entry_lookback, exit_lookback_long, exit_lookback_short
            )
            for asset in high.columns
        },
        index=high.index,
    )


# ---------------------------------------------------------------------------
# 2.7 EMA crossover — matches course notes Strategy B (Rohrbach et al.)
# ---------------------------------------------------------------------------
def ema_crossover_momentum(prices: pd.DataFrame, fast_span: int, slow_span: int) -> pd.DataFrame:
    """Fast/slow EMA crossover trend signal.

    +1 while the fast EMA is above the slow EMA (uptrend), -1 while below
    (downtrend), 0 wherever either EMA is still warming up (insufficient
    history). Per Rohrbach et al., "Momentum and Trend Following Trading
    Strategies for Currencies Revisited — Combining Academia and Industry".
    """
    fast_ema = prices.ewm(span=fast_span, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_span, adjust=False).mean()
    signal = np.sign(fast_ema - slow_ema)

    warmup = max(fast_span, slow_span)
    signal.iloc[: warmup - 1] = 0.0
    return signal


def blended_ema_crossover_momentum(
    prices: pd.DataFrame,
    spans: tuple[tuple[int, int], ...] = ((8, 24), (16, 48), (32, 96)),
) -> pd.DataFrame:
    """Average crossover signal across multiple fast/slow horizon pairs.

    Rohrbach et al. combine several EMA horizon pairs rather than trading a
    single one; the default spans (8/24, 16/48, 32/96) are the pairs listed
    in the course notes. Result ranges continuously from -1 to +1 (e.g. 0.33
    means 1 of the 3 horizon pairs is in an uptrend).
    """
    signals = [ema_crossover_momentum(prices, fast, slow) for fast, slow in spans]
    return sum(signals) / len(signals)
