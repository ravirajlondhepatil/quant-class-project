"""Tests for src/quant_project/signals/momentum.py (Feature 2).

Expected values in these tests are computed by hand (see comments), not by
calling the function under test with different inputs — the point is to
catch the function producing the wrong numbers, not just confirm it runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_project.signals.momentum import (
    activity_weighted_momentum,
    blended_ema_crossover_momentum,
    channel_breakout_momentum,
    ema_crossover_momentum,
    flow_window_momentum,
    seasonality_momentum,
    theme_momentum,
    time_horizon_momentum,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D")


# ---------------------------------------------------------------------------
# 2.1 time_horizon_momentum
# ---------------------------------------------------------------------------
def test_time_horizon_momentum_matches_hand_computed_pct_change():
    # A: +10% each step (100 -> 110 -> 121); B: -10% each step (50 -> 45 -> 40.5)
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 45.0, 40.5]}, index=_idx(3))

    lookback_1 = time_horizon_momentum(prices, 1)
    expected_1 = pd.DataFrame(
        {"A": [np.nan, 0.10, 0.10], "B": [np.nan, -0.10, -0.10]}, index=_idx(3)
    )
    pd.testing.assert_frame_equal(lookback_1, expected_1, check_exact=False, atol=1e-9)

    lookback_2 = time_horizon_momentum(prices, 2)
    expected_2 = pd.DataFrame(
        {"A": [np.nan, np.nan, 0.21], "B": [np.nan, np.nan, -0.19]}, index=_idx(3)
    )
    pd.testing.assert_frame_equal(lookback_2, expected_2, check_exact=False, atol=1e-9)


# ---------------------------------------------------------------------------
# 2.2 activity_weighted_momentum
# ---------------------------------------------------------------------------
def test_activity_weighted_momentum_scales_by_relative_activity():
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 108.9], "B": [50.0, 45.0, 40.5, 44.55]}, index=_idx(4)
    )
    activity = pd.DataFrame(
        {"A": [10.0, 20.0, 10.0, 10.0], "B": [10.0, 10.0, 10.0, 20.0]}, index=_idx(4)
    )

    result = activity_weighted_momentum(prices, activity, return_lookback=1, activity_lookback=2)

    # raw_momentum: A=[nan,.10,.10,-.10]  B=[nan,-.10,-.10,.10]
    # avg_activity (rolling(2).mean()): A=[nan,15,15,10]  B=[nan,10,10,15]
    # relative_activity = activity/avg_activity: A=[nan,4/3,2/3,1]  B=[nan,1,1,4/3]
    expected = pd.DataFrame(
        {
            "A": [np.nan, 0.10 * (20 / 15), 0.10 * (10 / 15), -0.10 * 1.0],
            "B": [np.nan, -0.10 * 1.0, -0.10 * 1.0, 0.10 * (20 / 15)],
        },
        index=_idx(4),
    )
    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=1e-9)


# ---------------------------------------------------------------------------
# 2.3 seasonality_momentum
# ---------------------------------------------------------------------------
def test_seasonality_momentum_averages_prior_occurrences_of_same_label():
    # labels: weekday, weekend, weekday, weekend
    prices = pd.DataFrame(
        {"A": [100.0, 200.0, 150.0, 180.0], "B": [100.0, 50.0, 25.0, 50.0]}, index=_idx(4)
    )
    labels = pd.Series(["weekday", "weekend", "weekday", "weekend"], index=_idx(4))

    result = seasonality_momentum(prices, labels, lookback_sessions=2)

    # A returns: [nan, 1.0, -0.25, 0.2]
    #   weekday subsequence (idx0,2): [nan, -0.25] -> rolling(2,min_periods=1): [nan, -0.25]
    #   weekend subsequence (idx1,3): [1.0, 0.2]   -> rolling(2,min_periods=1): [1.0, 0.6]
    # B returns: [nan, -0.5, -0.5, 1.0]
    #   weekday subsequence (idx0,2): [nan, -0.5]  -> rolling: [nan, -0.5]
    #   weekend subsequence (idx1,3): [-0.5, 1.0]  -> rolling: [-0.5, 0.25]
    expected = pd.DataFrame(
        {"A": [np.nan, 1.0, -0.25, 0.6], "B": [np.nan, -0.5, -0.5, 0.25]}, index=_idx(4)
    )
    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=1e-9)


# ---------------------------------------------------------------------------
# 2.4 theme_momentum
# ---------------------------------------------------------------------------
def test_theme_momentum_averages_peer_returns_and_nans_lone_themes():
    prices = pd.DataFrame(
        {"A": [100.0, 110.0], "B": [50.0, 60.0], "C": [10.0, 11.0]}, index=_idx(2)
    )
    theme_map = {"A": "L1", "B": "L1", "C": "DeFi"}

    result = theme_momentum(prices, theme_map, lookback=1)

    # A's only peer is B (return 0.20); B's only peer is A (return 0.10);
    # C has no peers in its theme -> all-NaN column.
    expected = pd.DataFrame(
        {"A": [np.nan, 0.20], "B": [np.nan, 0.10], "C": [np.nan, np.nan]}, index=_idx(2)
    )
    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=1e-9)


# ---------------------------------------------------------------------------
# 2.5 flow_window_momentum
# ---------------------------------------------------------------------------
def test_flow_window_momentum_averages_last_n_window_occurrences_not_calendar_bars():
    # Constructed so returns at the True (window) bars are 0.2, -0.1, 0.3 while
    # the False bars in between have different (irrelevant) returns — this
    # would expose the earlier bug where rolling ran over calendar bars
    # instead of over occurrences of the window.
    prices = pd.DataFrame({"A": [100.0, 120.0, 114.0, 102.6, 107.73, 140.049]}, index=_idx(6))
    window_mask = pd.Series([False, True, False, True, False, True], index=_idx(6))

    result = flow_window_momentum(prices, window_mask, lookback_occurrences=2)

    # window returns in order of occurrence: idx1=0.2, idx3=-0.1, idx5=0.3
    # rolling(2, min_periods=1) over that occurrence-subsequence:
    #   idx1: mean([0.2]) = 0.2
    #   idx3: mean([0.2, -0.1]) = 0.05
    #   idx5: mean([-0.1, 0.3]) = 0.1
    expected = pd.DataFrame({"A": [np.nan, 0.2, np.nan, 0.05, np.nan, 0.1]}, index=_idx(6))
    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=1e-6)


# ---------------------------------------------------------------------------
# 2.6 channel_breakout_momentum (Donchian-style — course notes Strategy A)
# ---------------------------------------------------------------------------
def test_channel_breakout_momentum_long_then_short_then_flat():
    # Hand-traced state machine with entry_lookback=3, exit_lookback_long=2,
    # exit_lookback_short=2 — see the accompanying derivation in the PR/commit
    # description for the full trace.
    high = pd.Series([10, 10, 10, 15, 12, 11, 13, 12, 12, 12], index=_idx(10), dtype=float)
    low = pd.Series([9, 9, 9, 10, 8, 7, 9, 8, 8, 9], index=_idx(10), dtype=float)

    result = channel_breakout_momentum(
        high.to_frame("X"),
        low.to_frame("X"),
        entry_lookback=3,
        exit_lookback_long=2,
        exit_lookback_short=2,
    )

    expected = pd.Series([0, 0, 0, 1, 0, -1, 0, 0, 0, 0], index=_idx(10), name="X")
    pd.testing.assert_series_equal(result["X"], expected, check_dtype=False)


def test_channel_breakout_momentum_rejects_mismatched_columns():
    high = pd.DataFrame({"A": [1.0, 2.0]}, index=_idx(2))
    low = pd.DataFrame({"B": [1.0, 2.0]}, index=_idx(2))
    with pytest.raises(ValueError, match="same columns"):
        channel_breakout_momentum(
            high, low, entry_lookback=1, exit_lookback_long=1, exit_lookback_short=1
        )


# ---------------------------------------------------------------------------
# 2.7 ema_crossover_momentum / blended_ema_crossover_momentum (Rohrbach et al.)
# ---------------------------------------------------------------------------
def test_ema_crossover_momentum_matches_sign_of_fast_minus_slow_after_warmup():
    prices = pd.DataFrame(
        {"A": [100.0, 102, 101, 105, 108, 110, 115, 120, 125, 130]}, index=_idx(10)
    )
    fast_span, slow_span = 2, 4

    result = ema_crossover_momentum(prices, fast_span, slow_span)

    fast_ema = prices.ewm(span=fast_span, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_span, adjust=False).mean()
    expected = np.sign(fast_ema - slow_ema)
    expected.iloc[: max(fast_span, slow_span) - 1] = 0.0

    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=1e-9)


def test_ema_crossover_momentum_zero_during_warmup():
    prices = pd.DataFrame({"A": [100.0, 200.0, 50.0, 300.0]}, index=_idx(4))
    result = ema_crossover_momentum(prices, fast_span=2, slow_span=4)
    # warmup = max(2,4) - 1 = 3 bars forced to exactly 0.0
    assert (result["A"].iloc[:3] == 0.0).all()


def test_blended_ema_crossover_momentum_averages_component_signals():
    prices = pd.DataFrame(
        {"A": [100.0, 102, 101, 105, 108, 110, 115, 120, 125, 130]}, index=_idx(10)
    )
    spans = ((2, 4), (3, 6))

    result = blended_ema_crossover_momentum(prices, spans=spans)

    component_a = ema_crossover_momentum(prices, 2, 4)
    component_b = ema_crossover_momentum(prices, 3, 6)
    expected = (component_a + component_b) / 2

    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=1e-9)
