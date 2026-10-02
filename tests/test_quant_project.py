"""Tests for the quant_project package — combined into one file.

Covers: data.py and audit.py (Phase 1 / Feature 7), signals/momentum.py
(Feature 2), signals/reversal.py (Feature 3), costs.py (Feature 4.5),
backtest.py (Feature 4), combination.py (Feature 5), and performance.py
(Feature 6). Expected values are hand-computed in comments, not derived by
calling the function under test differently — the point is to catch wrong
numbers, not just confirm it runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_project.audit import log_run, read_audit_log
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
from quant_project.data import (
    OHLCV_COLUMNS,
    DataRequest,
    ValidationReport,
    build_close_price_panel,
    build_exchange_client,
    cache_path,
    fetch_ohlcv_from_exchange,
    is_cache_fresh,
    load_multi_exchange_universe,
    load_symbol_ohlcv,
    load_universe_ohlcv,
    read_cache,
    validate_ohlcv,
    write_cache,
)
from quant_project.model_selection import (
    generate_walk_forward_folds,
    select_best,
    slice_backtest_result,
    split_prices,
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
from quant_project.signals.indicators import (
    average_pairwise_correlation_indicator,
    realized_volatility_indicator,
    return_dispersion_indicator,
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


class _FakeExchange:
    """Test double for a ccxt exchange client. Returns one fixed page of
    OHLCV rows per call (in the order given), regardless of the `since` it's
    called with — pagination progress in fetch_ohlcv_from_exchange is driven
    purely by the timestamps in what's returned, exactly like a real
    exchange, so this is enough to exercise that logic without a network
    call. Tracks `calls` so a test can assert pagination stopped when it
    should have.
    """

    def __init__(self, pages: list[list[list]]):
        self._pages = list(pages)
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls += 1
        if not self._pages:
            return []
        return self._pages.pop(0)


# =============================================================================
# audit.py — Feature 7.4
# =============================================================================
def test_log_run_appends_record_with_timestamp(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    record = log_run("data_load", {"symbol": "BTC/USDT"}, {"rows": 10}, log_path=log_path)
    assert record["event_type"] == "data_load"
    assert record["parameters"] == {"symbol": "BTC/USDT"}
    assert record["result_summary"] == {"rows": 10}
    assert "timestamp" in record
    assert read_audit_log(log_path) == [record]


def test_log_run_appends_multiple_records_in_order(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    log_run("a", {}, log_path=log_path)
    log_run("b", {}, log_path=log_path)
    assert [r["event_type"] for r in read_audit_log(log_path)] == ["a", "b"]


def test_log_run_defaults_result_summary_to_empty_dict(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    log_run("data_load", {"symbol": "BTC/USDT"}, log_path=log_path)
    assert read_audit_log(log_path)[0]["result_summary"] == {}


def test_log_run_stringifies_non_json_native_values(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    log_run("data_load", {"start": pd.Timestamp("2024-01-01")}, log_path=log_path)
    assert read_audit_log(log_path)[0]["parameters"]["start"] == str(pd.Timestamp("2024-01-01"))


def test_read_audit_log_missing_file_returns_empty(tmp_path):
    assert read_audit_log(tmp_path / "nope.jsonl") == []


# =============================================================================
# data.py — Phase 1 & Feature 7 (data layer)
# =============================================================================


# ---------------------------------------------------------------------------
# 1.2 / 7.1 DataRequest validation
# ---------------------------------------------------------------------------
def test_data_request_normalizes_start_end_to_timestamps():
    request = DataRequest("binance", ["BTC/USDT"], "1d", "2024-01-01", "2024-02-01")
    assert request.start == pd.Timestamp("2024-01-01")
    assert request.end == pd.Timestamp("2024-02-01")


def test_data_request_rejects_empty_symbols():
    with pytest.raises(ValueError):
        DataRequest("binance", [], "1d", "2024-01-01", "2024-02-01")


def test_data_request_rejects_blank_symbol():
    with pytest.raises(ValueError):
        DataRequest("binance", ["BTC/USDT", "  "], "1d", "2024-01-01", "2024-02-01")


def test_data_request_rejects_unsupported_timeframe():
    with pytest.raises(ValueError):
        DataRequest("binance", ["BTC/USDT"], "7m", "2024-01-01", "2024-02-01")


def test_data_request_rejects_inverted_date_range():
    with pytest.raises(ValueError):
        DataRequest("binance", ["BTC/USDT"], "1d", "2024-02-01", "2024-01-01")


def test_data_request_rejects_blank_exchange_id():
    with pytest.raises(ValueError):
        DataRequest("  ", ["BTC/USDT"], "1d", "2024-01-01", "2024-02-01")


# ---------------------------------------------------------------------------
# 1.1 / 7.2 Exchange client & OHLCV retrieval
# ---------------------------------------------------------------------------
def test_build_exchange_client_enables_rate_limiting():
    exchange = build_exchange_client("binance")
    assert exchange.enableRateLimit is True


def test_build_exchange_client_rejects_unknown_exchange():
    with pytest.raises(ValueError):
        build_exchange_client("not_a_real_exchange")


def test_fetch_ohlcv_from_exchange_paginates_until_short_page():
    # page1 is full-length (== limit) so pagination continues; page2 is
    # shorter than limit, which is the "no more data" signal to stop.
    page1 = [[0, 1, 1, 1, 1, 1], [60_000, 1, 1, 1, 1, 1]]
    page2 = [[120_000, 1, 1, 1, 1, 1]]
    exchange = _FakeExchange([page1, page2])

    result = fetch_ohlcv_from_exchange(
        exchange,
        "BTC/USDT",
        "1m",
        start=pd.Timestamp(0, unit="ms"),
        end=pd.Timestamp(200_000, unit="ms"),
        limit=2,
    )

    assert exchange.calls == 2
    assert len(result) == 3
    assert list(result.columns) == OHLCV_COLUMNS
    assert result.index[0] == pd.Timestamp(0, unit="ms")


def test_fetch_ohlcv_from_exchange_stops_on_stalled_cursor():
    # Full-length page (== limit) so the "short page" stop wouldn't fire on
    # its own, but its last timestamp doesn't advance past the cursor that
    # produced it -- the stalled-cursor guard must stop pagination after one
    # call instead of re-requesting the same window forever.
    stale_page = [[100, 1, 1, 1, 1, 1], [300, 1, 1, 1, 1, 1], [500, 1, 1, 1, 1, 1]]
    exchange = _FakeExchange([stale_page, stale_page, stale_page])

    fetch_ohlcv_from_exchange(
        exchange,
        "BTC/USDT",
        "1m",
        start=pd.Timestamp(1000, unit="ms"),
        end=pd.Timestamp(999_999, unit="ms"),
        limit=3,
    )

    assert exchange.calls == 1


def test_fetch_ohlcv_from_exchange_empty_result_has_expected_columns():
    exchange = _FakeExchange([])
    result = fetch_ohlcv_from_exchange(
        exchange,
        "BTC/USDT",
        "1m",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-02"),
    )
    assert list(result.columns) == OHLCV_COLUMNS
    assert result.empty


# ---------------------------------------------------------------------------
# 1.3 / 1.4 / 7.3 Parquet cache
# ---------------------------------------------------------------------------
def test_cache_path_is_deterministic_and_sanitizes_symbol():
    cache_dir = Path("/tmp/quant-project-cache-test")
    path = cache_path("binance", "BTC/USDT", "1d", cache_dir=cache_dir)
    assert path == cache_dir / "binance" / "1d" / "BTC-USDT.parquet"
    assert cache_path("binance", "BTC/USDT", "1d", cache_dir=cache_dir) == path


def test_read_cache_missing_file_returns_none(tmp_path):
    assert read_cache(tmp_path / "nope.parquet") is None


def test_write_then_read_cache_roundtrips(tmp_path):
    df = pd.DataFrame({"open": [1.0, 2.0]}, index=_idx(2))
    path = tmp_path / "sub" / "BTC-USDT.parquet"
    write_cache(df, path)
    pd.testing.assert_frame_equal(read_cache(path), df, check_freq=False)


def test_is_cache_fresh_true_when_range_covered():
    idx = pd.date_range("2024-01-01", "2024-01-31", freq="D")
    cached = pd.DataFrame({"close": range(len(idx))}, index=idx)
    assert is_cache_fresh(cached, pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-20")) is True


def test_is_cache_fresh_false_when_range_not_covered():
    idx = pd.date_range("2024-01-01", "2024-01-10", freq="D")
    cached = pd.DataFrame({"close": range(len(idx))}, index=idx)
    assert is_cache_fresh(cached, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-20")) is False


def test_is_cache_fresh_false_when_none_or_empty():
    assert is_cache_fresh(None, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")) is False
    assert (
        is_cache_fresh(pd.DataFrame(), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))
        is False
    )


# ---------------------------------------------------------------------------
# 1.5 validate_ohlcv
# ---------------------------------------------------------------------------
def test_validate_ohlcv_removes_duplicate_timestamps():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])
    df = pd.DataFrame({"close": [1.0, 1.5, 2.0]}, index=idx)
    report = validate_ohlcv(df, "1d")
    assert report.duplicates_removed == 1
    assert list(report.cleaned["close"]) == [1.0, 2.0]  # keeps first occurrence


def test_validate_ohlcv_detects_gaps():
    idx = pd.date_range("2024-01-01", "2024-01-05", freq="D").delete(2)  # drop 01-03
    df = pd.DataFrame({"close": range(len(idx))}, index=idx)
    report = validate_ohlcv(df, "1d")
    assert list(report.gap_timestamps) == [pd.Timestamp("2024-01-03")]


def test_validate_ohlcv_flags_insufficient_history():
    df = pd.DataFrame({"close": range(5)}, index=pd.date_range("2024-01-01", periods=5, freq="D"))
    report = validate_ohlcv(df, "1d", min_periods=30)
    assert report.has_sufficient_history is False


def test_validate_ohlcv_sufficient_history_when_enough_bars():
    df = pd.DataFrame({"close": range(30)}, index=pd.date_range("2024-01-01", periods=30, freq="D"))
    report = validate_ohlcv(df, "1d", min_periods=30)
    assert report.has_sufficient_history is True


def test_validate_ohlcv_no_gaps_reported_for_short_series():
    df = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"]))
    report = validate_ohlcv(df, "1d")
    assert len(report.gap_timestamps) == 0


# ---------------------------------------------------------------------------
# load_symbol_ohlcv / load_universe_ohlcv — orchestration
# ---------------------------------------------------------------------------
def test_load_symbol_ohlcv_fetches_then_reuses_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    audit_log_path = tmp_path / "audit.jsonl"
    request = DataRequest(
        "binance", ["BTC/USDT"], "1m", pd.Timestamp(0, unit="ms"), pd.Timestamp(120_000, unit="ms")
    )
    page = [[0, 1, 1, 1, 1, 1], [60_000, 1, 1, 1, 1, 1], [120_000, 1, 1, 1, 1, 1]]
    exchange = _FakeExchange([page])

    df1, report1 = load_symbol_ohlcv(
        request, "BTC/USDT", cache_dir=cache_dir, exchange=exchange, audit_log_path=audit_log_path
    )
    assert len(df1) == 3
    assert exchange.calls == 1
    assert report1.duplicates_removed == 0

    # Cache now fully covers the requested range -> second call must not
    # trigger another exchange fetch.
    df2, _ = load_symbol_ohlcv(
        request, "BTC/USDT", cache_dir=cache_dir, exchange=exchange, audit_log_path=audit_log_path
    )
    pd.testing.assert_frame_equal(df2, df1)
    assert exchange.calls == 1

    records = read_audit_log(audit_log_path)
    assert [r["result_summary"]["source"] for r in records] == ["fetched", "cache"]


def test_load_symbol_ohlcv_force_refresh_bypasses_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    request = DataRequest(
        "binance", ["BTC/USDT"], "1m", pd.Timestamp(0, unit="ms"), pd.Timestamp(120_000, unit="ms")
    )
    page = [[0, 1, 1, 1, 1, 1], [60_000, 1, 1, 1, 1, 1], [120_000, 1, 1, 1, 1, 1]]
    exchange = _FakeExchange([page, page])

    load_symbol_ohlcv(
        request, "BTC/USDT", cache_dir=cache_dir, exchange=exchange, audit_log_path=None
    )
    assert exchange.calls == 1

    load_symbol_ohlcv(
        request,
        "BTC/USDT",
        cache_dir=cache_dir,
        exchange=exchange,
        force_refresh=True,
        audit_log_path=None,
    )
    assert exchange.calls == 2


def test_load_universe_ohlcv_loads_every_symbol(tmp_path):
    cache_dir = tmp_path / "cache"
    request = DataRequest(
        "binance",
        ["BTC/USDT", "ETH/USDT"],
        "1m",
        pd.Timestamp(0, unit="ms"),
        pd.Timestamp(60_000, unit="ms"),
    )
    page = [[0, 1, 1, 1, 1, 1], [60_000, 1, 1, 1, 1, 1]]
    exchange = _FakeExchange([list(page), list(page)])

    results = load_universe_ohlcv(
        request, cache_dir=cache_dir, exchange=exchange, audit_log_path=None
    )

    assert set(results.keys()) == {"BTC/USDT", "ETH/USDT"}
    for df, _report in results.values():
        assert len(df) == 2
    assert exchange.calls == 2


# ---------------------------------------------------------------------------
# 1.1 load_multi_exchange_universe — literal "multi-exchange" ingestion
# ---------------------------------------------------------------------------
def test_load_multi_exchange_universe_merges_across_exchanges(tmp_path):
    cache_dir = tmp_path / "cache"
    request_binance = DataRequest(
        "binance", ["BTC/USDT"], "1m", pd.Timestamp(0, unit="ms"), pd.Timestamp(60_000, unit="ms")
    )
    request_kraken = DataRequest(
        "kraken", ["ETH/USD"], "1m", pd.Timestamp(0, unit="ms"), pd.Timestamp(60_000, unit="ms")
    )
    page = [[0, 1, 1, 1, 1, 1], [60_000, 1, 1, 1, 1, 1]]
    exchange_binance = _FakeExchange([list(page)])
    exchange_kraken = _FakeExchange([list(page)])

    results = load_multi_exchange_universe(
        [request_binance, request_kraken],
        cache_dir=cache_dir,
        exchanges={"binance": exchange_binance, "kraken": exchange_kraken},
        audit_log_path=None,
    )

    assert set(results.keys()) == {"BTC/USDT", "ETH/USD"}
    assert exchange_binance.calls == 1
    assert exchange_kraken.calls == 1


def test_load_multi_exchange_universe_rejects_duplicate_symbol_across_exchanges(tmp_path):
    cache_dir = tmp_path / "cache"
    request_a = DataRequest(
        "binance", ["BTC/USDT"], "1m", pd.Timestamp(0, unit="ms"), pd.Timestamp(60_000, unit="ms")
    )
    request_b = DataRequest(
        "kraken", ["BTC/USDT"], "1m", pd.Timestamp(0, unit="ms"), pd.Timestamp(60_000, unit="ms")
    )
    page = [[0, 1, 1, 1, 1, 1], [60_000, 1, 1, 1, 1, 1]]
    exchange = _FakeExchange([list(page), list(page)])

    with pytest.raises(ValueError):
        load_multi_exchange_universe(
            [request_a, request_b],
            cache_dir=cache_dir,
            exchanges={"binance": exchange, "kraken": exchange},
            audit_log_path=None,
        )


# ---------------------------------------------------------------------------
# build_close_price_panel
# ---------------------------------------------------------------------------
def _universe_entry(closes: list[float], sufficient: bool = True) -> tuple:
    df = pd.DataFrame({"close": closes}, index=_idx(len(closes)))
    report = ValidationReport(
        cleaned=df,
        duplicates_removed=0,
        gap_timestamps=pd.DatetimeIndex([]),
        has_sufficient_history=sufficient,
    )
    return df, report


def test_build_close_price_panel_combines_symbols_into_wide_frame():
    universe = {
        "BTC/USDT": _universe_entry([100.0, 101.0]),
        "ETH/USDT": _universe_entry([10.0, 10.5]),
    }
    panel = build_close_price_panel(universe)
    expected = pd.DataFrame(
        {"BTC/USDT": [100.0, 101.0], "ETH/USDT": [10.0, 10.5]}, index=_idx(2)
    )
    pd.testing.assert_frame_equal(panel.sort_index(axis=1), expected.sort_index(axis=1))


def test_build_close_price_panel_drops_insufficient_history_by_default():
    universe = {
        "BTC/USDT": _universe_entry([100.0, 101.0], sufficient=True),
        "NEW/USDT": _universe_entry([1.0, 1.1], sufficient=False),
    }
    panel = build_close_price_panel(universe)
    assert list(panel.columns) == ["BTC/USDT"]


def test_build_close_price_panel_can_keep_insufficient_history():
    universe = {
        "BTC/USDT": _universe_entry([100.0, 101.0], sufficient=True),
        "NEW/USDT": _universe_entry([1.0, 1.1], sufficient=False),
    }
    panel = build_close_price_panel(universe, require_sufficient_history=False)
    assert set(panel.columns) == {"BTC/USDT", "NEW/USDT"}


def test_build_close_price_panel_rejects_all_insufficient():
    universe = {"NEW/USDT": _universe_entry([1.0, 1.1], sufficient=False)}
    with pytest.raises(ValueError):
        build_close_price_panel(universe)


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
def test_macro_conditioned_reversal_scales_by_expanding_dislocation_percentile():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1]}, index=_idx(4))
    # Not monotonic, specifically so an *expanding* percentile differs from
    # ranking against the whole series (the B3 bug this replaces): at t=1,
    # 1.0 is the lowest value seen *so far* (50th percentile of {3.0, 1.0}),
    # but it's also the lowest of the *whole* series -- under the old bug
    # it would have ranked 0.25 there instead of 0.5.
    dislocation = pd.Series([3.0, 1.0, 4.0, 2.0], index=_idx(4))

    result = macro_conditioned_reversal(
        prices, lookback=1, dislocation_indicator=dislocation, dislocation_threshold_quantile=0.7
    )

    # reversal = -pct_change(1) = [nan, -.10, -.10, -.10]
    # expanding percentile at t = (count of values seen through t that are
    # <= value[t]) / (count of values seen through t):
    #   t0: {3.0} -> 1/1 = 1.0
    #   t1: {3.0, 1.0}, value=1.0 -> 1/2 = 0.5
    #   t2: {3.0, 1.0, 4.0}, value=4.0 -> 3/3 = 1.0
    #   t3: {3.0, 1.0, 4.0, 2.0}, value=2.0 -> 2/4 = 0.5
    # active (>=0.7): [T, F, T, F] -> scale = [1.0, 0, 1.0, 0]
    expected = pd.DataFrame({"A": [np.nan, -0.0, -0.10, -0.0]}, index=_idx(4))
    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=1e-9)


def test_macro_conditioned_reversal_handles_nan_indicator_values():
    # A leading NaN in the indicator (e.g. a rolling-computed indicator's
    # own warm-up, exactly what signals/indicators.py's functions produce)
    # must be skipped, not propagate into every later percentile.
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=_idx(3))
    dislocation = pd.Series([np.nan, 1.0, 2.0], index=_idx(3))

    result = macro_conditioned_reversal(
        prices, lookback=1, dislocation_indicator=dislocation, dislocation_threshold_quantile=0.7
    )

    # reversal = -pct_change(1) = [nan, -.10, -.10]
    # percentile_rank: t0 undefined (NaN input) -> NaN;
    #   t1: valid={1.0} -> 1/1 = 1.0 ; t2: valid={1.0, 2.0} -> 2/2 = 1.0
    # active (>=0.7): NaN>=0.7 is False -> [F, T, T] -> scale = [0.0, 1.0, 1.0]
    expected = pd.DataFrame({"A": [np.nan, -0.10, -0.10]}, index=_idx(3))
    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=1e-9)


# ---------------------------------------------------------------------------
# signals/indicators.py — macro dislocation indicators for 3.4
# ---------------------------------------------------------------------------
def test_realized_volatility_indicator_averages_rolling_asset_vol():
    # A: 100 -> 110 -> 99 (+10%, -10% exactly); B: 100 -> 90 -> 99 (-10%, +10%)
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0], "B": [100.0, 90.0, 99.0]}, index=_idx(3))

    result = realized_volatility_indicator(prices, lookback=2)

    # returns: A=[nan,.10,-.10], B=[nan,-.10,.10]
    # rolling(2).std() needs 2 non-NaN in the window:
    #   t0: NaN (only the NaN return so far) ; t1: window has 1 valid value -> NaN
    #   t2: window=[.10,-.10] (or [-.10,.10]) -> std (ddof=1) = sqrt(0.02) for both assets
    # mean across assets at t2 = sqrt(0.02)
    expected = pd.Series([np.nan, np.nan, 0.02**0.5], index=_idx(3))
    pd.testing.assert_series_equal(result, expected, atol=1e-9)


def test_return_dispersion_indicator_instantaneous_cross_sectional_std():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0], "B": [100.0, 90.0, 99.0]}, index=_idx(3))

    result = return_dispersion_indicator(prices)

    # t0: both returns NaN -> NaN
    # t1: [.10, -.10] -> std (ddof=1) = sqrt(0.02) ; t2: [-.10, .10] -> same
    expected = pd.Series([np.nan, 0.02**0.5, 0.02**0.5], index=_idx(3))
    pd.testing.assert_series_equal(result, expected, atol=1e-9)


def test_return_dispersion_indicator_lookback_smooths_with_rolling_mean():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0], "B": [100.0, 90.0, 99.0]}, index=_idx(3))

    result = return_dispersion_indicator(prices, lookback=2)

    # instantaneous dispersion = [nan, sqrt(0.02), sqrt(0.02)]
    # rolling(2).mean(): t1's window [nan, sqrt(0.02)] has only 1 valid -> NaN
    #                    t2's window [sqrt(0.02), sqrt(0.02)] -> sqrt(0.02)
    expected = pd.Series([np.nan, np.nan, 0.02**0.5], index=_idx(3))
    pd.testing.assert_series_equal(result, expected, atol=1e-9)


def test_average_pairwise_correlation_indicator_perfectly_anticorrelated():
    # A and B move exactly opposite each other every period -> corr = -1.0
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0], "B": [100.0, 90.0, 99.0]}, index=_idx(3))

    result = average_pairwise_correlation_indicator(prices, lookback=2)

    # t0: before lookback window is available -> NaN
    # t1: window=[t0(nan,nan), t1(.10,-.10)] -> only 1 valid row -> corr undefined -> NaN
    # t2: window=[t1(.10,-.10), t2(-.10,.10)] -> exactly anti-correlated -> -1.0
    expected = pd.Series([np.nan, np.nan, -1.0], index=_idx(3))
    pd.testing.assert_series_equal(result, expected, atol=1e-9)


def test_average_pairwise_correlation_indicator_rejects_single_asset():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=_idx(3))
    with pytest.raises(ValueError):
        average_pairwise_correlation_indicator(prices, lookback=2)


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
# 5.2 inverse_vol_weights (time-varying per specs/methodology-fixes-scope.md B1)
# ---------------------------------------------------------------------------
def test_inverse_vol_weights_is_time_varying_and_favors_steadier_strategy():
    # returns_b is always exactly 1/10th the magnitude of returns_a, so
    # whatever expanding window is used, b's trailing std is always 1/10th
    # of a's -- the same 1/11 : 10/11 split should hold at every row once
    # there's enough strictly-past history, not just once over the full
    # sample.
    idx = _idx(5)
    returns_a = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03], index=idx)
    returns_b = returns_a * 0.1

    weights = inverse_vol_weights({"a": returns_a, "b": returns_b}, min_periods=2)

    # Rows 0-1: fewer than 2 strictly-past (shift(1)'d) observations exist
    # yet -- undefined.
    assert weights.iloc[0].isna().all()
    assert weights.iloc[1].isna().all()
    for t in range(2, 5):
        assert weights["a"].iloc[t] == pytest.approx(1 / 11)
        assert weights["b"].iloc[t] == pytest.approx(10 / 11)
        assert weights.iloc[t].sum() == pytest.approx(1.0)


def test_inverse_vol_weights_excludes_bar_ts_own_return():
    # B1: a weight set for bar t must only reflect returns known *before*
    # t. Changing bar 4's own return must not change bar 4's weight (nor
    # any earlier bar's).
    idx = _idx(5)
    returns_a = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03], index=idx)
    returns_b = returns_a * 0.1
    weights_1 = inverse_vol_weights({"a": returns_a, "b": returns_b}, min_periods=2)

    returns_a_alt = returns_a.copy()
    returns_a_alt.iloc[4] = 999.0
    weights_2 = inverse_vol_weights({"a": returns_a_alt, "b": returns_b}, min_periods=2)

    pd.testing.assert_frame_equal(weights_1, weights_2)


def test_inverse_vol_weights_treats_zero_trailing_vol_as_undefined_not_an_error():
    # A strategy whose trailing return history has zero variance would
    # divide by zero if taken at face value; it should drop out as
    # undefined for that bar (letting the other strategy take 100% of the
    # weight) rather than raising and aborting the whole run.
    idx = _idx(4)
    flat = pd.Series([0.01, 0.01, 0.01, 0.01], index=idx)
    normal = pd.Series([0.01, -0.02, 0.03, -0.01], index=idx)

    weights = inverse_vol_weights({"flat": flat, "normal": normal}, min_periods=2)

    assert weights["flat"].iloc[2:].isna().all()
    assert (weights["normal"].iloc[2:] == 1.0).all()


def test_inverse_vol_weights_rejects_empty():
    with pytest.raises(ValueError):
        inverse_vol_weights({})


# ---------------------------------------------------------------------------
# 5.3 ic_weights (time-varying + genuinely-forward return, B1 + B2)
# ---------------------------------------------------------------------------
def test_ic_weights_is_time_varying_and_uses_genuinely_forward_returns():
    # 3 assets, 4 identical periods (so the rank-correlation math is the
    # same every period -- only the expanding/shift machinery varies).
    # "perfect": signal ranks match return ranks exactly -> per-period IC=1
    # "partial": signal ranks [1,3,2] vs return ranks [1,2,3]
    #   -> rank diffs (0,1,-1), rho = 1 - 6*sum(d^2)/(n(n^2-1)) = 1 - 12/24 = 0.5
    idx = _idx(4)
    forward_returns = pd.DataFrame(
        {"X": [0.01] * 4, "Y": [0.02] * 4, "Z": [0.03] * 4}, index=idx
    )
    signals = {
        "perfect": pd.DataFrame(
            {"X": [10.0] * 4, "Y": [20.0] * 4, "Z": [30.0] * 4}, index=idx
        ),
        "partial": pd.DataFrame(
            {"X": [10.0] * 4, "Y": [30.0] * 4, "Z": [20.0] * 4}, index=idx
        ),
    }

    weights = ic_weights(signals, forward_returns, min_periods=1)

    # Row 0: no strictly-past, fully-realized IC exists yet (the IC
    # attributed to row 0 itself needs row 1's forward return, so it isn't
    # knowable until row 1) -- undefined.
    assert weights.iloc[0].isna().all()
    # Rows 1-3: weight_perfect = 1.0/(1.0+0.5) = 2/3 ; weight_partial = 1/3,
    # at every row (the per-period IC is identical every period here).
    for t in range(1, 4):
        assert weights["perfect"].iloc[t] == pytest.approx(2 / 3)
        assert weights["partial"].iloc[t] == pytest.approx(1 / 3)


def test_ic_weights_weight_at_t_unaffected_by_returns_strictly_after_t():
    # B2/B1 together: a weight set for bar t must not depend on returns
    # realized after t. Scrambling only the *last* period's return must
    # leave every earlier bar's weight unchanged.
    idx = _idx(5)
    forward_returns = pd.DataFrame(
        {"X": [0.01] * 5, "Y": [0.02] * 5, "Z": [0.03] * 5}, index=idx
    )
    signal = pd.DataFrame(
        {"X": [10.0] * 5, "Y": [20.0] * 5, "Z": [30.0] * 5}, index=idx
    )

    weights_1 = ic_weights({"s": signal}, forward_returns, min_periods=1)

    forward_returns_alt = forward_returns.copy()
    forward_returns_alt.iloc[4] = [999.0, -999.0, 0.0]
    weights_2 = ic_weights({"s": signal}, forward_returns_alt, min_periods=1)

    pd.testing.assert_series_equal(weights_1["s"].iloc[:4], weights_2["s"].iloc[:4])


def test_ic_weights_floors_negative_ic_at_zero():
    # "inverted": signal ranks are the exact opposite of return ranks ->
    # per-period IC=-1, floored to 0 and excluded entirely once renormalized.
    idx = _idx(3)
    forward_returns = pd.DataFrame(
        {"X": [0.01] * 3, "Y": [0.02] * 3, "Z": [0.03] * 3}, index=idx
    )
    signals = {
        "perfect": pd.DataFrame(
            {"X": [10.0] * 3, "Y": [20.0] * 3, "Z": [30.0] * 3}, index=idx
        ),
        "inverted": pd.DataFrame(
            {"X": [30.0] * 3, "Y": [20.0] * 3, "Z": [10.0] * 3}, index=idx
        ),
    }

    weights = ic_weights(signals, forward_returns, min_periods=1)

    assert weights["perfect"].iloc[1:].eq(1.0).all()
    assert weights["inverted"].iloc[1:].eq(0.0).all()


def test_ic_weights_rejects_empty():
    with pytest.raises(ValueError):
        ic_weights({}, pd.DataFrame())


def test_ic_weights_undefined_when_no_strategy_has_positive_trailing_ic():
    # Replaces the old "rejects all-negative IC" behavior: with only one
    # (always-negative-IC) strategy, every row's weight is undefined
    # (NaN) rather than the call raising -- a multi-year backtest
    # shouldn't abort over a stretch where nothing currently looks good.
    idx = _idx(3)
    forward_returns = pd.DataFrame(
        {"X": [0.01] * 3, "Y": [0.02] * 3, "Z": [0.03] * 3}, index=idx
    )
    signals = {
        "inverted": pd.DataFrame(
            {"X": [30.0] * 3, "Y": [20.0] * 3, "Z": [10.0] * 3}, index=idx
        ),
    }

    weights = ic_weights(signals, forward_returns, min_periods=1)

    assert weights["inverted"].isna().all()


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
    idx = _idx(3)
    signal_a = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0]}, index=idx)
    signal_b = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0]}, index=idx)
    returns_a = pd.Series([0.01, -0.01, 0.02], index=idx)
    returns_b = returns_a * 0.1

    combined = combine_signals(
        {"a": signal_a, "b": signal_b},
        method="inverse_vol",
        strategy_returns={"a": returns_a, "b": returns_b},
        min_periods=2,
    )

    # Rows 0-1: both strategies' weights are undefined (not enough
    # strictly-past history yet) -> the combined row stays NaN. Row 2: both
    # signals are identical, so they standardize to the same row (z-score
    # of [1, -1] is [1/sqrt(2), -1/sqrt(2)]); whatever the weight split
    # between "a" and "b" turns out to be, the combined row is still that
    # same value since the weights sum to 1.
    z = 1 / (2**0.5)
    expected = pd.DataFrame(
        {"A": [np.nan, np.nan, z], "B": [np.nan, np.nan, -z]}, index=idx
    )
    pd.testing.assert_frame_equal(combined, expected)


def test_combine_signals_dispatches_to_ic_weights():
    idx = _idx(4)
    forward_returns = pd.DataFrame(
        {"X": [0.01] * 4, "Y": [0.02] * 4, "Z": [0.03] * 4}, index=idx
    )
    signal_a = pd.DataFrame(
        {"X": [10.0] * 4, "Y": [20.0] * 4, "Z": [30.0] * 4}, index=idx
    )
    signal_b = pd.DataFrame(
        {"X": [10.0] * 4, "Y": [30.0] * 4, "Z": [20.0] * 4}, index=idx
    )

    combined = combine_signals(
        {"a": signal_a, "b": signal_b},
        method="ic",
        forward_returns=forward_returns,
        min_periods=1,
    )

    expected_weights = ic_weights(
        {"a": signal_a, "b": signal_b}, forward_returns, min_periods=1
    )
    expected = standardize_signal(signal_a).mul(expected_weights["a"], axis=0).add(
        standardize_signal(signal_b).mul(expected_weights["b"], axis=0), fill_value=0.0
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
def test_sharpe_ratio_mean_over_std_formula():
    # C1: mean(returns) / std(returns) * sqrt(periods_per_year) -- not the
    # geometric annualized-return / annualized-vol figure this used to be.
    returns = pd.Series([0.06, -0.04], index=_idx(2))
    # mean = 0.01 ; std (ddof=1) = sqrt(((0.05)^2 + (-0.05)^2)/1) = sqrt(0.005)
    # sharpe = 0.01/sqrt(0.005) * sqrt(2) = 0.01*sqrt(0.005^-1 * 2) = 0.01*sqrt(400) = 0.01*20 = 0.2
    result = sharpe_ratio(returns, periods_per_year=2)
    assert result == pytest.approx(0.2)


def test_sharpe_ratio_subtracts_deannualized_risk_free_rate():
    returns = pd.Series([0.06, -0.04], index=_idx(2))
    # per-period risk-free = 0.04 / 2 = 0.02 ; excess mean = 0.01 - 0.02 = -0.01
    # sharpe = -0.01/sqrt(0.005) * sqrt(2) = -0.01*20 = -0.2
    result = sharpe_ratio(returns, periods_per_year=2, risk_free_rate=0.04)
    assert result == pytest.approx(-0.2)


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
    # recover beta=1.5 and a per-period alpha of 0.002 exactly. Residual
    # variance is only ~0 up to floating-point roundoff (not exactly 0), so
    # alpha's t-stat comes out enormous rather than NaN -- this is the
    # "fit is as perfect as floating point allows" case, not the "residual
    # variance is a true, exact zero" edge case (that's covered by the
    # zero-standard-error branch directly, below). Correlation is exactly
    # 1.0 either way (perfect fit, positive slope).
    benchmark = pd.Series([0.01, 0.02, -0.01, 0.03], index=_idx(4))
    strategy = 1.5 * benchmark + 0.002

    result = alpha_beta(strategy, benchmark, periods_per_year=1)
    assert result.beta == pytest.approx(1.5)
    assert result.alpha == pytest.approx(0.002)  # periods_per_year=1 -> no compounding effect
    assert abs(result.alpha_t_stat) > 1e6
    assert result.correlation == pytest.approx(1.0)


def test_alpha_beta_alpha_t_stat_nan_when_standard_error_is_exactly_zero():
    # Force the alpha_se == 0.0 branch directly (rather than relying on
    # floating-point roundoff happening to land on exactly 0) by using
    # integer-valued inputs an exact line fits through with zero residual.
    benchmark = pd.Series([1.0, 2.0, 3.0], index=_idx(3))
    strategy = pd.Series([2.0, 4.0, 6.0], index=_idx(3))  # = 2 * benchmark exactly

    result = alpha_beta(strategy, benchmark, periods_per_year=1)
    assert result.beta == pytest.approx(2.0)
    assert result.alpha == pytest.approx(0.0)
    assert np.isnan(result.alpha_t_stat)


def test_alpha_beta_computes_t_stat_and_correlation_with_noise():
    # Not a perfect fit, so there's real residual variance to compute a
    # finite t-stat from.
    benchmark = pd.Series([1.0, 2.0, 3.0, 4.0], index=_idx(4))
    strategy = pd.Series([2.0, 3.0, 5.0, 6.0], index=_idx(4))

    result = alpha_beta(strategy, benchmark, periods_per_year=1)

    # cov(x,y) [ddof=1] = 7/3 ; var(x) = 5/3 -> beta = 7/5 = 1.4
    # alpha_per_period = ybar(4.0) - beta*xbar(2.5) = 4.0 - 3.5 = 0.5 (ppy=1)
    # fitted = 0.5 + 1.4*x -> residuals = [0.1, -0.3, 0.3, -0.1]
    #   RSS = 0.01+0.09+0.09+0.01 = 0.20 ; residual_var = 0.20/(4-2) = 0.10
    # alpha_se = sqrt(0.10 * (1/4 + 2.5^2/5.0)) = sqrt(0.10*1.5) = sqrt(0.15) = sqrt(15)/10
    # t_stat = 0.5 / (sqrt(15)/10) = 5/sqrt(15) = sqrt(15)/3
    # corr = cov/sqrt(var_x*var_y) = (7/3)/sqrt((5/3)*(10/3)) = 7/sqrt(50) = 7*sqrt(2)/10
    assert result.beta == pytest.approx(1.4)
    assert result.alpha == pytest.approx(0.5)
    assert result.alpha_t_stat == pytest.approx(15**0.5 / 3)
    assert result.correlation == pytest.approx(7 * 2**0.5 / 10)


def test_alpha_beta_rejects_zero_variance_benchmark():
    benchmark = pd.Series([0.01, 0.01, 0.01], index=_idx(3))
    strategy = pd.Series([0.02, -0.01, 0.03], index=_idx(3))
    with pytest.raises(ValueError):
        alpha_beta(strategy, benchmark)


def test_alpha_beta_rejects_insufficient_overlap():
    benchmark = pd.Series([0.01, 0.02], index=_idx(2))
    strategy = pd.Series([0.02, 0.03], index=_idx(2))
    with pytest.raises(ValueError):
        alpha_beta(strategy, benchmark)


# ---------------------------------------------------------------------------
# 6.6 / 6.7 build_performance_report
# ---------------------------------------------------------------------------
def test_build_performance_report_assembles_all_metrics():
    net_returns = pd.Series([0.05, -0.05, 0.02], index=_idx(3))
    gross_returns = pd.Series([0.06, -0.04, 0.03], index=_idx(3))
    result = BacktestResult(
        weights=pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=_idx(3)),
        turnover=pd.Series([1.0, 0.0, 0.0], index=_idx(3)),
        gross_returns=gross_returns,
        net_returns=net_returns,
    )
    # Benchmark identical to net_returns -> beta=1.0, alpha=0.0, corr=1.0 exactly.
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
    assert np.isnan(report.alpha_t_stat)  # zero residual variance -> undefined t-stat
    assert report.correlation == pytest.approx(1.0)


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
    assert report.alpha_t_stat is None
    assert report.correlation is None


# =============================================================================
# model_selection.py — C3/C4
# =============================================================================


# ---------------------------------------------------------------------------
# split_prices
# ---------------------------------------------------------------------------
def test_split_prices_splits_chronologically_by_fraction():
    idx = _idx(10)
    prices = pd.DataFrame({"A": range(10)}, index=idx)
    split = split_prices(prices, train_fraction=0.7)

    assert len(split.train_prices) == 7
    assert len(split.test_prices) == 3
    assert split.split_date == idx[7]
    assert list(split.train_prices.index) == list(idx[:7])
    assert list(split.test_prices.index) == list(idx[7:])


def test_split_prices_never_overlaps():
    idx = _idx(10)
    prices = pd.DataFrame({"A": range(10)}, index=idx)
    split = split_prices(prices, train_fraction=0.5)
    assert split.train_prices.index.max() < split.test_prices.index.min()


def test_split_prices_clamps_to_at_least_one_bar_each_side():
    idx = _idx(3)
    prices = pd.DataFrame({"A": range(3)}, index=idx)
    # train_fraction=0.01 would naively round down to 0 train bars -> clamp to 1.
    split = split_prices(prices, train_fraction=0.01)
    assert len(split.train_prices) == 1
    assert len(split.test_prices) == 2

    # train_fraction=0.99 would naively round up to all 3 bars in train,
    # leaving none for test -> clamp to leave at least 1 test bar.
    split = split_prices(prices, train_fraction=0.99)
    assert len(split.train_prices) == 2
    assert len(split.test_prices) == 1


def test_split_prices_rejects_invalid_train_fraction():
    prices = pd.DataFrame({"A": range(5)}, index=_idx(5))
    with pytest.raises(ValueError):
        split_prices(prices, train_fraction=0.0)
    with pytest.raises(ValueError):
        split_prices(prices, train_fraction=1.0)


def test_split_prices_rejects_too_few_bars():
    prices = pd.DataFrame({"A": [1.0]}, index=_idx(1))
    with pytest.raises(ValueError):
        split_prices(prices, train_fraction=0.5)


# ---------------------------------------------------------------------------
# generate_walk_forward_folds
# ---------------------------------------------------------------------------
def test_generate_walk_forward_folds_expanding_window_and_full_coverage():
    # n=20, initial_train_fraction=0.5 -> initial_train_end=10, remaining=10,
    # split into 4 folds -> boundaries [12, 15, 18, 20] (hand-verified via
    # Python's round(), which is round-half-to-even).
    idx = _idx(20)
    prices = pd.DataFrame({"A": range(20)}, index=idx)

    folds = generate_walk_forward_folds(prices, n_folds=4, initial_train_fraction=0.5)

    assert len(folds) == 4
    expected_train_len = [10, 12, 15, 18]
    expected_test_len = [2, 3, 3, 2]
    for i, fold in enumerate(folds):
        assert len(fold.train_prices) == expected_train_len[i]
        assert len(fold.test_prices) == expected_test_len[i]
        # Expanding: each fold's train window is exactly the previous
        # fold's train window plus its test window (anchored, not rolling).
        if i > 0:
            assert list(fold.train_prices.index) == list(folds[i - 1].train_prices.index) + list(
                folds[i - 1].test_prices.index
            )

    # No bar dropped or duplicated: every fold's test block end to end
    # reconstructs exactly the remaining (post-initial-train) history.
    reconstructed = pd.concat([fold.test_prices for fold in folds])
    pd.testing.assert_index_equal(reconstructed.index, idx[10:])


def test_generate_walk_forward_folds_rejects_invalid_inputs():
    prices = pd.DataFrame({"A": range(20)}, index=_idx(20))
    with pytest.raises(ValueError):
        generate_walk_forward_folds(prices, initial_train_fraction=0.0)
    with pytest.raises(ValueError):
        generate_walk_forward_folds(prices, initial_train_fraction=1.0)
    with pytest.raises(ValueError):
        generate_walk_forward_folds(prices, n_folds=0)


def test_generate_walk_forward_folds_rejects_too_many_folds_for_remaining_bars():
    prices = pd.DataFrame({"A": range(10)}, index=_idx(10))
    # initial_train_fraction=0.9 -> only 1 bar remains, can't make 5 folds from it.
    with pytest.raises(ValueError):
        generate_walk_forward_folds(prices, n_folds=5, initial_train_fraction=0.9)


# ---------------------------------------------------------------------------
# select_best
# ---------------------------------------------------------------------------
def test_select_best_picks_highest_score():
    assert select_best({"a": 1.0, "b": 2.5, "c": -1.0}) == "b"


def test_select_best_skips_nan_scores():
    assert select_best({"a": np.nan, "b": 0.5}) == "b"


def test_select_best_rejects_all_nan():
    with pytest.raises(ValueError):
        select_best({"a": np.nan, "b": np.nan})


def test_select_best_rejects_empty():
    with pytest.raises(ValueError):
        select_best({})


# ---------------------------------------------------------------------------
# slice_backtest_result
# ---------------------------------------------------------------------------
def test_slice_backtest_result_restricts_every_field():
    idx = _idx(5)
    result = BacktestResult(
        weights=pd.DataFrame({"A": range(5), "B": range(5, 10)}, index=idx),
        turnover=pd.Series(range(5), index=idx, dtype=float),
        gross_returns=pd.Series(range(5), index=idx, dtype=float),
        net_returns=pd.Series(range(5), index=idx, dtype=float),
    )
    sub_index = idx[2:4]

    sliced = slice_backtest_result(result, sub_index)

    pd.testing.assert_frame_equal(sliced.weights, result.weights.loc[sub_index])
    pd.testing.assert_series_equal(sliced.turnover, result.turnover.loc[sub_index])
    pd.testing.assert_series_equal(sliced.gross_returns, result.gross_returns.loc[sub_index])
    pd.testing.assert_series_equal(sliced.net_returns, result.net_returns.loc[sub_index])
