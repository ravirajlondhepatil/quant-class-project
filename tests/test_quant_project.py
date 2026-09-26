"""Tests for the quant_project package — combined into one file.

Covers: signals/momentum.py (Feature 2), signals/reversal.py (Feature 3),
costs.py (Feature 4.5), and backtest.py (Feature 4). Expected values are
hand-computed in comments, not derived by calling the function under test
differently — the point is to catch wrong numbers, not just confirm it runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_project.backtest import (
    BacktestResult,
    UnconstrainedBacktester,
    build_dollar_neutral_weights,
    run_multi_strategy_backtest,
)
from quant_project.combination import (
    combine_signals,
    equal_weights,
    ic_weights,
    inverse_vol_weights,
    standardize_signal,
)
from quant_project.costs import (
    LIMIT_ORDER_COST_BPS,
    MARKET_ORDER_COST_BPS,
    apply_transaction_costs,
    cost_bps_for,
    net_of_costs,
)
from quant_project.performance import (
    alpha_beta,
    annualized_return,
    annualized_volatility,
    build_performance_report,
    cumulative_returns,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
)
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
from quant_project.signals.reversal import (
    correlation_reversal,
    macro_conditioned_reversal,
    pairs_trading_distance_signal,
    time_horizon_reversal,
    uninformed_reversal,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D")


# =============================================================================
# signals/momentum.py (Feature 2)
# =============================================================================


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


# =============================================================================
# signals/reversal.py (Feature 3)
# =============================================================================


# ---------------------------------------------------------------------------
# 3.1 time_horizon_reversal
# ---------------------------------------------------------------------------
def test_time_horizon_reversal_is_negative_of_pct_change():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 45.0, 40.5]}, index=_idx(3))

    result = time_horizon_reversal(prices, lookback=1)

    expected = pd.DataFrame({"A": [np.nan, -0.10, -0.10], "B": [np.nan, 0.10, 0.10]}, index=_idx(3))
    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=1e-9)


# ---------------------------------------------------------------------------
# 3.2 uninformed_reversal
# ---------------------------------------------------------------------------
def test_uninformed_reversal_masks_out_high_activity_assets():
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0], "B": [100.0, 90.0, 81.0], "C": [100.0, 105.0, 110.25]},
        index=_idx(3),
    )
    activity = pd.DataFrame(
        {"A": [10.0, 10.0, 30.0], "B": [10.0, 10.0, 10.0], "C": [10.0, 10.0, 20.0]},
        index=_idx(3),
    )

    result = uninformed_reversal(
        prices, activity, lookback=1, activity_lookback=2, low_activity_quantile=0.5
    )

    # base reversal (=-pct_change): A=[nan,-.10,-.10] B=[nan,.10,.10] C=[nan,-.05,-.05]
    # relative_activity = activity / rolling(2).mean():
    #   t1: A=10/10=1.0  B=10/10=1.0  C=10/10=1.0  -> median=1.0 -> all <=1.0 -> all kept
    #   t2: A=30/20=1.5  B=10/10=1.0  C=20/15=1.333 -> median=1.333 -> only B,C <=1.333 -> A masked
    expected = pd.DataFrame(
        {
            "A": [np.nan, -0.10, np.nan],
            "B": [np.nan, 0.10, 0.10],
            "C": [np.nan, -0.05, -0.05],
        },
        index=_idx(3),
    )
    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=1e-9)


# ---------------------------------------------------------------------------
# 3.3 correlation_reversal
# ---------------------------------------------------------------------------
def test_correlation_reversal_scores_opposite_signs_for_a_pair():
    # B is flat (0 return always); A alternates +10%/-10%/+10% so the A-B
    # return spread has non-degenerate rolling variance to z-score against.
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 99.0, 108.9], "B": [100.0, 100.0, 100.0, 100.0]}, index=_idx(4)
    )

    result = correlation_reversal(prices, pairs=[("A", "B")], lookback=1, zscore_window=2)

    # spread (A ret - B ret) = [nan, 0.10, -0.10, 0.10]
    # rolling(2) mean/std: t2 -> mean=0.0, std=sqrt(((.1)^2+(-.1)^2)/1)=0.14142
    #                       t3 -> mean=0.0, std=0.14142 (same magnitude, opposite order)
    # spread_z: t2 = -0.10/0.14142 = -0.7071   t3 = 0.10/0.14142 = 0.7071
    # signal[A] = -spread_z, signal[B] = +spread_z
    expected = pd.DataFrame(
        {
            "A": [np.nan, np.nan, 0.7071, -0.7071],
            "B": [np.nan, np.nan, -0.7071, 0.7071],
        },
        index=_idx(4),
    )
    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=1e-3)


# ---------------------------------------------------------------------------
# 3.4 macro_conditioned_reversal
# ---------------------------------------------------------------------------
def test_macro_conditioned_reversal_scales_by_dislocation_percentile():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1]}, index=_idx(4))
    dislocation = pd.Series([1.0, 2.0, 3.0, 4.0], index=_idx(4))  # monotonic -> clean percentiles

    result = macro_conditioned_reversal(
        prices, lookback=1, dislocation_indicator=dislocation, dislocation_threshold_quantile=0.7
    )

    # reversal = -pct_change(1) = [nan, -.10, -.10, -.10]
    # percentile_rank (pct=True, 4 ascending values) = [.25, .50, .75, 1.0]
    # active (>=0.7): [F, F, T, T] -> scale = [0, 0, .75, 1.0]
    # result = reversal * scale
    expected = pd.DataFrame({"A": [np.nan, -0.0, -0.075, -0.10]}, index=_idx(4))
    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=1e-9)


# ---------------------------------------------------------------------------
# 3.5 pairs_trading_distance_signal (Gatev, Goetzmann & Rouwenhorst, 2006)
# ---------------------------------------------------------------------------
def test_pairs_trading_distance_signal_opens_on_divergence_closes_on_convergence():
    # B held flat so the normalized A/B spread tracks A's cumulative return
    # over the formation window exactly -- easy to hand-trace.
    a = pd.Series([100.0, 100.0, 110.0, 121.0, 108.9, 108.9], index=_idx(6))
    b = pd.Series([100.0] * 6, index=_idx(6))
    prices = pd.DataFrame({"A": a, "B": b})

    result = pairs_trading_distance_signal(
        prices, pairs=[("A", "B")], formation_window=2, entry_z=1.0, exit_z=0.5
    )

    # norm_a = A / A.shift(2): [nan, nan, 1.10, 1.21, 0.99, 0.90]; norm_b = 1 (after warmup)
    # spread = norm_a - 1: [nan, nan, 0.10, 0.21, -0.01, -0.10]
    # spread.rolling(2).std() (sample std): idx0,1 NaN (insufficient);
    #   idx2: only 1 of 2 window values non-NaN -> NaN (min_periods=window=2 by default)
    #   idx3: std([0.10,0.21]) = 0.07778   idx4: std([0.21,-0.01]) = 0.15556
    #   idx5: std([-0.01,-0.10]) = 0.06364
    # state machine (entry_z=1.0, exit_z=0.5), flat until std available:
    #   t3: spread=0.21 >= 1.0*0.07778 -> A outperformed -> short A/long B (-1)
    #   t4: |spread|=0.01 <= 0.5*0.15556=0.0778 -> converge -> flat (0)
    #   t5: spread=-0.10 <= -1.0*0.06364=-0.06364 -> A underperformed -> long A/short B (+1)
    expected_a = pd.Series([0, 0, 0, -1, 0, 1], index=_idx(6), name="A")
    expected_b = pd.Series([0, 0, 0, 1, 0, -1], index=_idx(6), name="B")
    pd.testing.assert_series_equal(result["A"], expected_a, check_dtype=False)
    pd.testing.assert_series_equal(result["B"], expected_b, check_dtype=False)


# =============================================================================
# costs.py (Feature 4.5)
# =============================================================================
def test_cost_bps_for_market_is_commission_plus_slippage():
    assert cost_bps_for("market") == 20.0 == MARKET_ORDER_COST_BPS


def test_cost_bps_for_limit_is_commission_only():
    assert cost_bps_for("limit") == 7.0 == LIMIT_ORDER_COST_BPS


def test_cost_bps_for_rejects_unknown_order_type():
    with pytest.raises(ValueError, match="Unknown order_type"):
        cost_bps_for("stop")  # type: ignore[arg-type]


def test_apply_transaction_costs_scales_turnover_by_cost_rate():
    turnover = pd.Series([0.0, 0.5, 1.0])

    market_cost = apply_transaction_costs(turnover, order_type="market")
    pd.testing.assert_series_equal(market_cost, pd.Series([0.0, 0.0010, 0.0020]))

    limit_cost = apply_transaction_costs(turnover, order_type="limit")
    pd.testing.assert_series_equal(limit_cost, pd.Series([0.0, 0.00035, 0.0007]))


def test_net_of_costs_subtracts_cost_from_gross_returns():
    gross_returns = pd.Series([0.01, 0.02])
    turnover = pd.Series([0.5, 1.0])

    net = net_of_costs(gross_returns, turnover, order_type="market")

    # market cost = 20bps * turnover: [0.001, 0.002]
    pd.testing.assert_series_equal(net, pd.Series([0.009, 0.018]), check_exact=False, atol=1e-9)


# =============================================================================
# backtest.py (Feature 4)
# =============================================================================


# ---------------------------------------------------------------------------
# build_dollar_neutral_weights
# ---------------------------------------------------------------------------
def test_build_dollar_neutral_weights_demeans_and_scales_to_unit_gross():
    signal = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]})

    weights = build_dollar_neutral_weights(signal)

    # mean=2 -> demeaned=[-1,0,1] -> gross=2 -> weights=[-0.5,0,0.5]
    expected = pd.DataFrame({"A": [-0.5], "B": [0.0], "C": [0.5]})
    pd.testing.assert_frame_equal(weights, expected, check_exact=False, atol=1e-9)


def test_build_dollar_neutral_weights_nans_zero_dispersion_rows():
    # All three assets have the same signal value -> no long/short spread
    # to build a position from -> NaN, not a divide-by-zero crash.
    signal = pd.DataFrame({"A": [5.0], "B": [5.0], "C": [5.0]})

    weights = build_dollar_neutral_weights(signal)

    assert weights.isna().all(axis=None)


# ---------------------------------------------------------------------------
# UnconstrainedBacktester — core mechanics
# ---------------------------------------------------------------------------
def test_backtester_lags_positions_turnover_and_net_returns():
    # Signal at t is traded into a position that earns the return over
    # t -> t+1, i.e. positions are the *prior* bar's weights.
    signal = pd.DataFrame({"A": [1.0, 2.0, -1.0, 0.0], "B": [-1.0, -2.0, 1.0, 0.0]}, index=_idx(4))
    returns = pd.DataFrame(
        {"A": [0.10, -0.10, 0.20, 0.05], "B": [0.20, 0.10, -0.10, 0.05]}, index=_idx(4)
    )

    result = UnconstrainedBacktester(signal, returns, order_type="market").run()

    # raw dollar-neutral weights per bar (2-asset case always splits +-0.5):
    #   t0: [0.5,-0.5]  t1: [0.5,-0.5]  t2: [-0.5,0.5]  t3: NaN (zero dispersion)
    # positions = weights shifted by 1, NaN/missing -> 0 (flat):
    #   t0: [0,0]  t1: [0.5,-0.5]  t2: [0.5,-0.5]  t3: [-0.5,0.5]
    expected_positions = pd.DataFrame(
        {"A": [0.0, 0.5, 0.5, -0.5], "B": [0.0, -0.5, -0.5, 0.5]}, index=_idx(4)
    )
    pd.testing.assert_frame_equal(result.weights, expected_positions, check_exact=False, atol=1e-9)

    # gross_returns = sum(position * return):
    #   t0: 0            t1: 0.5*-0.10 + -0.5*0.10 = -0.10
    #   t2: 0.5*0.20 + -0.5*-0.10 = 0.15   t3: -0.5*0.05 + 0.5*0.05 = 0
    expected_gross = pd.Series([0.0, -0.10, 0.15, 0.0], index=_idx(4))
    pd.testing.assert_series_equal(
        result.gross_returns, expected_gross, check_exact=False, atol=1e-9
    )

    # turnover = |position change| summed across assets (first bar vs flat):
    #   t0: |0|+|0| = 0        t1: |0.5-0|+|-0.5-0| = 1.0
    #   t2: |0.5-0.5|+|-0.5--0.5| = 0     t3: |-0.5-0.5|+|0.5--0.5| = 2.0
    expected_turnover = pd.Series([0.0, 1.0, 0.0, 2.0], index=_idx(4))
    pd.testing.assert_series_equal(result.turnover, expected_turnover, check_exact=False, atol=1e-9)

    # net_returns = gross - turnover * 20bps:
    #   t0: 0   t1: -0.10 - 0.002 = -0.102   t2: 0.15   t3: 0 - 0.004 = -0.004
    expected_net = pd.Series([0.0, -0.102, 0.15, -0.004], index=_idx(4))
    pd.testing.assert_series_equal(result.net_returns, expected_net, check_exact=False, atol=1e-9)


def test_backtester_limit_orders_cost_less_than_market_orders():
    signal = pd.DataFrame({"A": [1.0, -1.0], "B": [-1.0, 1.0]}, index=_idx(2))
    returns = pd.DataFrame({"A": [0.0, 0.10], "B": [0.0, -0.10]}, index=_idx(2))

    market_result = UnconstrainedBacktester(signal, returns, order_type="market").run()
    limit_result = UnconstrainedBacktester(signal, returns, order_type="limit").run()

    # Same gross returns and turnover, but limit (7bps) drags less than
    # market (20bps) -> limit net returns are (weakly) higher.
    pd.testing.assert_series_equal(market_result.gross_returns, limit_result.gross_returns)
    pd.testing.assert_series_equal(market_result.turnover, limit_result.turnover)
    assert (limit_result.net_returns >= market_result.net_returns).all()
    assert (limit_result.net_returns > market_result.net_returns).any()


def test_backtester_rejects_invalid_rebalance_every():
    signal = pd.DataFrame({"A": [1.0], "B": [-1.0]})
    returns = pd.DataFrame({"A": [0.0], "B": [0.0]})
    with pytest.raises(ValueError, match="rebalance_every"):
        UnconstrainedBacktester(signal, returns, rebalance_every=0)


def test_backtester_date_range_slices_inputs():
    signal = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0, -1.0]}, index=_idx(4))
    returns = pd.DataFrame({"A": [0.0] * 4, "B": [0.0] * 4}, index=_idx(4))

    result = UnconstrainedBacktester(signal, returns, start=_idx(4)[1], end=_idx(4)[2]).run()

    pd.testing.assert_index_equal(result.weights.index, _idx(4)[1:3])


# ---------------------------------------------------------------------------
# rebalance_every — hold weights between rebalances
# ---------------------------------------------------------------------------
def test_rebalance_every_holds_weight_until_next_rebalance():
    # 3 assets so weight *ratios* (not just sign) differ bar to bar, making a
    # "held vs. fresh" mixup visible. Rebalance bars are t0 and t2
    # (index % 2 == 0); t1 and t3's own signals are constructed differently
    # from t0/t2 specifically so a bug that used them would change the result.
    signal = pd.DataFrame(
        {
            "A": [3.0, 1.0, 2.0, 0.0],
            "B": [1.0, 3.0, -2.0, 0.0],
            "C": [-4.0, -4.0, 0.0, 0.0],
        },
        index=_idx(4),
    )
    returns = pd.DataFrame({"A": [0.0] * 4, "B": [0.0] * 4, "C": [0.0] * 4}, index=_idx(4))

    result = UnconstrainedBacktester(signal, returns, rebalance_every=2).run()

    # t0 (rebalance): mean=0, demeaned=[3,1,-4], gross=8 -> w0=[.375,.125,-.5]
    # t1 (held, ignores its own signal [1,3,-4]) -> stays w0
    # t2 (rebalance): mean=0, demeaned=[2,-2,0], gross=4 -> w2=[.5,-.5,0]
    # t3 (held) -> stays w2
    # positions = held weights shifted by 1 (first bar flat):
    expected_positions = pd.DataFrame(
        {
            "A": [0.0, 0.375, 0.375, 0.5],
            "B": [0.0, 0.125, 0.125, -0.5],
            "C": [0.0, -0.5, -0.5, 0.0],
        },
        index=_idx(4),
    )
    pd.testing.assert_frame_equal(result.weights, expected_positions, check_exact=False, atol=1e-9)

    # No trade should happen going from t1 -> t2's *position* (both hold w0)
    assert result.turnover.iloc[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_multi_strategy_backtest — Feature 4.3
# ---------------------------------------------------------------------------
def test_run_multi_strategy_backtest_matches_individual_runs():
    signal_a = pd.DataFrame({"A": [1.0, -1.0], "B": [-1.0, 1.0]}, index=_idx(2))
    signal_b = pd.DataFrame({"A": [2.0, 2.0], "B": [-2.0, -2.0]}, index=_idx(2))
    returns = pd.DataFrame({"A": [0.05, -0.05], "B": [-0.05, 0.05]}, index=_idx(2))

    results = run_multi_strategy_backtest({"a": signal_a, "b": signal_b}, returns)

    expected_a = UnconstrainedBacktester(signal_a, returns).run()
    expected_b = UnconstrainedBacktester(signal_b, returns).run()

    pd.testing.assert_series_equal(results["a"].net_returns, expected_a.net_returns)
    pd.testing.assert_series_equal(results["b"].net_returns, expected_b.net_returns)


# =============================================================================
# combination.py — Feature 5
# =============================================================================


# ---------------------------------------------------------------------------
# standardize_signal
# ---------------------------------------------------------------------------
def test_standardize_signal_row_zscore():
    signal = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]}, index=_idx(1))
    # mean=2, std (ddof=1) = sqrt(((1)^2+0+(1)^2)/2) = 1 -> z = [-1, 0, 1]
    expected = pd.DataFrame({"A": [-1.0], "B": [0.0], "C": [1.0]}, index=_idx(1))
    pd.testing.assert_frame_equal(standardize_signal(signal), expected)


def test_standardize_signal_zero_dispersion_row_is_nan():
    signal = pd.DataFrame({"A": [5.0], "B": [5.0]}, index=_idx(1))
    result = standardize_signal(signal)
    assert result.isna().all(axis=None)


# ---------------------------------------------------------------------------
# 5.1 equal_weights
# ---------------------------------------------------------------------------
def test_equal_weights_splits_evenly():
    weights = equal_weights(["a", "b", "c"])
    pd.testing.assert_series_equal(weights, pd.Series([1 / 3, 1 / 3, 1 / 3], index=["a", "b", "c"]))


def test_equal_weights_rejects_empty():
    with pytest.raises(ValueError):
        equal_weights([])


# ---------------------------------------------------------------------------
# 5.2 inverse_vol_weights
# ---------------------------------------------------------------------------
def test_inverse_vol_weights_favors_steadier_strategy():
    # std_a = sqrt(((0.01)^2*2 + (0.02)^2*2)/3) = sqrt(0.001/3) -> call it x
    # std_b = std of returns scaled by 0.1 -> x/10
    # weight_a = (1/x) / (1/x + 10/x) = 1/11 ; weight_b = 10/11
    returns_a = pd.Series([0.01, -0.01, 0.02, -0.02], index=_idx(4))
    returns_b = returns_a * 0.1
    weights = inverse_vol_weights({"a": returns_a, "b": returns_b})
    assert weights["a"] == pytest.approx(1 / 11)
    assert weights["b"] == pytest.approx(10 / 11)
    assert weights.sum() == pytest.approx(1.0)


def test_inverse_vol_weights_rejects_zero_vol_strategy():
    returns_flat = pd.Series([0.01, 0.01, 0.01], index=_idx(3))
    returns_normal = pd.Series([0.01, -0.02, 0.03], index=_idx(3))
    with pytest.raises(ValueError):
        inverse_vol_weights({"flat": returns_flat, "normal": returns_normal})


def test_inverse_vol_weights_rejects_empty():
    with pytest.raises(ValueError):
        inverse_vol_weights({})


# ---------------------------------------------------------------------------
# 5.3 ic_weights
# ---------------------------------------------------------------------------
def test_ic_weights_scores_by_average_rank_correlation():
    # 3 assets, 2 identical periods.
    # "perfect": signal ranks [1,2,3] match return ranks [1,2,3] exactly -> IC=1
    # "partial": signal ranks [1,3,2] vs return ranks [1,2,3]
    #   -> rank diffs (0,1,-1), rho = 1 - 6*sum(d^2)/(n(n^2-1)) = 1 - 12/24 = 0.5
    # weight_perfect = 1.0 / (1.0 + 0.5) = 2/3 ; weight_partial = 0.5/1.5 = 1/3
    forward_returns = pd.DataFrame(
        {"X": [0.01, 0.01], "Y": [0.02, 0.02], "Z": [0.03, 0.03]}, index=_idx(2)
    )
    signals = {
        "perfect": pd.DataFrame(
            {"X": [10.0, 10.0], "Y": [20.0, 20.0], "Z": [30.0, 30.0]}, index=_idx(2)
        ),
        "partial": pd.DataFrame(
            {"X": [10.0, 10.0], "Y": [30.0, 30.0], "Z": [20.0, 20.0]}, index=_idx(2)
        ),
    }
    weights = ic_weights(signals, forward_returns)
    assert weights["perfect"] == pytest.approx(2 / 3)
    assert weights["partial"] == pytest.approx(1 / 3)
    assert weights.sum() == pytest.approx(1.0)


def test_ic_weights_floors_negative_ic_at_zero():
    # "inverted": signal ranks are the exact opposite of return ranks -> IC=-1,
    # floored to 0 and excluded entirely once renormalized.
    forward_returns = pd.DataFrame(
        {"X": [0.01, 0.01], "Y": [0.02, 0.02], "Z": [0.03, 0.03]}, index=_idx(2)
    )
    signals = {
        "perfect": pd.DataFrame(
            {"X": [10.0, 10.0], "Y": [20.0, 20.0], "Z": [30.0, 30.0]}, index=_idx(2)
        ),
        "inverted": pd.DataFrame(
            {"X": [30.0, 30.0], "Y": [20.0, 20.0], "Z": [10.0, 10.0]}, index=_idx(2)
        ),
    }
    weights = ic_weights(signals, forward_returns)
    assert weights["perfect"] == pytest.approx(1.0)
    assert weights["inverted"] == pytest.approx(0.0)


def test_ic_weights_rejects_empty():
    with pytest.raises(ValueError):
        ic_weights({}, pd.DataFrame())


def test_ic_weights_rejects_all_negative_ic():
    forward_returns = pd.DataFrame(
        {"X": [0.01, 0.01], "Y": [0.02, 0.02], "Z": [0.03, 0.03]}, index=_idx(2)
    )
    signals = {
        "inverted": pd.DataFrame(
            {"X": [30.0, 30.0], "Y": [20.0, 20.0], "Z": [10.0, 10.0]}, index=_idx(2)
        ),
    }
    with pytest.raises(ValueError):
        ic_weights(signals, forward_returns)


# ---------------------------------------------------------------------------
# 5.4 combine_signals — configurable method selection
# ---------------------------------------------------------------------------
def test_combine_signals_equal_weight_averages_standardized_signals():
    signal_a = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]}, index=_idx(1))
    signal_b = pd.DataFrame({"A": [3.0], "B": [2.0], "C": [1.0]}, index=_idx(1))
    # standardize_signal(a) = [-1, 0, 1] ; standardize_signal(b) = [1, 0, -1]
    # equal-weighted average -> [0, 0, 0]
    combined = combine_signals({"a": signal_a, "b": signal_b}, method="equal")
    expected = pd.DataFrame({"A": [0.0], "B": [0.0], "C": [0.0]}, index=_idx(1))
    pd.testing.assert_frame_equal(combined, expected)


def test_combine_signals_dispatches_to_inverse_vol_weights():
    signal_a = pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=_idx(1))
    signal_b = pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=_idx(1))
    returns_a = pd.Series([0.01, -0.01, 0.02, -0.02], index=_idx(4))
    returns_b = returns_a * 0.1

    combined = combine_signals(
        {"a": signal_a, "b": signal_b},
        method="inverse_vol",
        strategy_returns={"a": returns_a, "b": returns_b},
    )
    # Both signals are identical, so they standardize to the same row
    # (z-score of [1, -1] is [1/sqrt(2), -1/sqrt(2)]); whatever the weight
    # split between "a" and "b", the combined row is still that same value
    # since the weights sum to 1.
    z = 1 / (2**0.5)
    expected = pd.DataFrame({"A": [z], "B": [-z]}, index=_idx(1))
    pd.testing.assert_frame_equal(combined, expected)


def test_combine_signals_dispatches_to_ic_weights():
    forward_returns = pd.DataFrame(
        {"X": [0.01, 0.01], "Y": [0.02, 0.02], "Z": [0.03, 0.03]}, index=_idx(2)
    )
    signal_a = pd.DataFrame(
        {"X": [10.0, 10.0], "Y": [20.0, 20.0], "Z": [30.0, 30.0]}, index=_idx(2)
    )
    signal_b = pd.DataFrame(
        {"X": [30.0, 30.0], "Y": [10.0, 10.0], "Z": [20.0, 20.0]}, index=_idx(2)
    )
    combined = combine_signals(
        {"a": signal_a, "b": signal_b}, method="ic", forward_returns=forward_returns
    )
    expected_weights = ic_weights({"a": signal_a, "b": signal_b}, forward_returns)
    expected = (
        standardize_signal(signal_a) * expected_weights["a"]
        + standardize_signal(signal_b) * expected_weights["b"]
    )
    pd.testing.assert_frame_equal(combined, expected)


def test_combine_signals_requires_supporting_data_for_method():
    signal = pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=_idx(1))
    with pytest.raises(ValueError):
        combine_signals({"a": signal}, method="inverse_vol")
    with pytest.raises(ValueError):
        combine_signals({"a": signal}, method="ic")


def test_combine_signals_rejects_unknown_method():
    signal = pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=_idx(1))
    with pytest.raises(ValueError):
        combine_signals({"a": signal}, method="not_a_real_method")


def test_combine_signals_rejects_empty():
    with pytest.raises(ValueError):
        combine_signals({}, method="equal")


# =============================================================================
# performance.py — Feature 6
# =============================================================================


# ---------------------------------------------------------------------------
# 6.1 cumulative_returns
# ---------------------------------------------------------------------------
def test_cumulative_returns_compounds():
    returns = pd.Series([0.1, -0.1, 0.1], index=_idx(3))
    # step1: 1.1 - 1 = 0.1
    # step2: 1.1*0.9 = 0.99 -> -0.01
    # step3: 0.99*1.1 = 1.089 -> 0.089
    expected = pd.Series([0.1, -0.01, 0.089], index=_idx(3))
    pd.testing.assert_series_equal(cumulative_returns(returns), expected, atol=1e-9)


def test_cumulative_returns_treats_nan_as_zero():
    returns = pd.Series([0.1, np.nan, 0.1], index=_idx(3))
    # step2 is a no-op (treated as 0 return): stays at 0.1
    # step3: 1.1*1.1 = 1.21 -> 0.21
    expected = pd.Series([0.1, 0.1, 0.21], index=_idx(3))
    pd.testing.assert_series_equal(cumulative_returns(returns), expected, atol=1e-9)


# ---------------------------------------------------------------------------
# 6.2 annualized_return / annualized_volatility
# ---------------------------------------------------------------------------
def test_annualized_return_compounds_and_scales_by_periods_ratio():
    returns = pd.Series([0.1, 0.1], index=_idx(2))
    # total_growth = 1.1*1.1 = 1.21 ; periods_per_year/n_periods = 4/2 = 2
    # annualized = 1.21^2 - 1 = 1.4641 - 1 = 0.4641
    result = annualized_return(returns, periods_per_year=4)
    assert result == pytest.approx(0.4641)


def test_annualized_return_empty_series_is_nan():
    assert np.isnan(annualized_return(pd.Series([], dtype=float)))


def test_annualized_volatility_scales_std_by_sqrt_periods():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02], index=_idx(4))
    # var (ddof=1) = (0.0001+0.0001+0.0004+0.0004)/3 = 0.001/3
    expected = (0.001 / 3) ** 0.5 * 2  # sqrt(periods_per_year=4) = 2
    result = annualized_volatility(returns, periods_per_year=4)
    assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 6.3 sharpe_ratio
# ---------------------------------------------------------------------------
def test_sharpe_ratio_combines_annualized_return_and_vol():
    returns = pd.Series([0.05, -0.05], index=_idx(2))
    # total_growth = 1.05*0.95 = 0.9975 -> annualized_return (n=2, ppy=2) = -0.0025
    # var (ddof=1) = (0.05^2+0.05^2)/1 = 0.005 -> annualized_vol = sqrt(0.005*2) = sqrt(0.01) = 0.1
    # sharpe = -0.0025 / 0.1 = -0.025
    result = sharpe_ratio(returns, periods_per_year=2)
    assert result == pytest.approx(-0.025)


def test_sharpe_ratio_nan_when_vol_is_zero():
    returns = pd.Series([0.02, 0.02], index=_idx(2))
    assert np.isnan(sharpe_ratio(returns))


# ---------------------------------------------------------------------------
# 6.4 / 6.6 drawdown_series / max_drawdown
# ---------------------------------------------------------------------------
def test_drawdown_series_tracks_peak_to_trough():
    returns = pd.Series([0.1, -0.2, 0.05], index=_idx(3))
    # wealth = [1.1, 0.88, 0.924] ; running max = [1.1, 1.1, 1.1]
    # drawdown = [0, 0.88/1.1 - 1, 0.924/1.1 - 1] = [0, -0.2, -0.16]
    expected = pd.Series([0.0, -0.2, -0.16], index=_idx(3))
    pd.testing.assert_series_equal(drawdown_series(returns), expected, atol=1e-9)


def test_max_drawdown_is_series_minimum():
    returns = pd.Series([0.1, -0.2, 0.05], index=_idx(3))
    assert max_drawdown(returns) == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# 6.5 alpha_beta
# ---------------------------------------------------------------------------
def test_alpha_beta_recovers_exact_linear_relationship():
    # strategy = 1.5 * benchmark + 0.002 exactly (no noise), so OLS should
    # recover beta=1.5 and a per-period alpha of 0.002 exactly.
    benchmark = pd.Series([0.01, 0.02, -0.01, 0.03], index=_idx(4))
    strategy = 1.5 * benchmark + 0.002

    alpha, beta = alpha_beta(strategy, benchmark, periods_per_year=1)
    assert beta == pytest.approx(1.5)
    assert alpha == pytest.approx(0.002)  # periods_per_year=1 -> no compounding effect


def test_alpha_beta_rejects_zero_variance_benchmark():
    benchmark = pd.Series([0.01, 0.01, 0.01], index=_idx(3))
    strategy = pd.Series([0.02, -0.01, 0.03], index=_idx(3))
    with pytest.raises(ValueError):
        alpha_beta(strategy, benchmark)


def test_alpha_beta_rejects_insufficient_overlap():
    benchmark = pd.Series([0.01], index=_idx(1))
    strategy = pd.Series([0.02], index=_idx(1))
    with pytest.raises(ValueError):
        alpha_beta(strategy, benchmark)


# ---------------------------------------------------------------------------
# 6.6 / 6.7 build_performance_report
# ---------------------------------------------------------------------------
def test_build_performance_report_assembles_all_metrics():
    net_returns = pd.Series([0.05, -0.05], index=_idx(2))
    gross_returns = pd.Series([0.06, -0.04], index=_idx(2))
    result = BacktestResult(
        weights=pd.DataFrame({"A": [1.0, 1.0]}, index=_idx(2)),
        turnover=pd.Series([1.0, 0.0], index=_idx(2)),
        gross_returns=gross_returns,
        net_returns=net_returns,
    )
    # Benchmark identical to net_returns -> beta=1.0, alpha=0.0 exactly.
    benchmark = net_returns.copy()

    report = build_performance_report(result, periods_per_year=2, benchmark_returns=benchmark)

    pd.testing.assert_series_equal(report.cumulative_gross, cumulative_returns(gross_returns))
    pd.testing.assert_series_equal(report.cumulative_net, cumulative_returns(net_returns))
    pd.testing.assert_series_equal(report.drawdown, drawdown_series(net_returns))
    assert report.annualized_return == pytest.approx(annualized_return(net_returns, 2))
    assert report.annualized_volatility == pytest.approx(annualized_volatility(net_returns, 2))
    assert report.sharpe_ratio == pytest.approx(sharpe_ratio(net_returns, 2))
    assert report.max_drawdown == pytest.approx(max_drawdown(net_returns))
    assert report.alpha == pytest.approx(0.0)
    assert report.beta == pytest.approx(1.0)


def test_build_performance_report_without_benchmark_leaves_alpha_beta_none():
    net_returns = pd.Series([0.01, 0.02], index=_idx(2))
    result = BacktestResult(
        weights=pd.DataFrame({"A": [1.0, 1.0]}, index=_idx(2)),
        turnover=pd.Series([0.0, 0.0], index=_idx(2)),
        gross_returns=net_returns,
        net_returns=net_returns,
    )
    report = build_performance_report(result)
    assert report.alpha is None
    assert report.beta is None
