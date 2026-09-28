"""Macro dislocation indicators for ``reversal.macro_conditioned_reversal``.

ref/ClassProject.docx names four indicators to test as the "volatility /
dislocation in the macro environment" driver of reversal: implied
volatility, realized volatility, return dispersion, and average pairwise
correlation. ``macro_conditioned_reversal`` takes any of these as a generic
``dislocation_indicator`` time series; this module computes three of them
directly from the OHLCV price panel this project already ingests.

Implied volatility is *not* implemented here — it comes from an options
market (an IV surface), which isn't derivable from the spot OHLCV
``data.py`` fetches. Wiring that in would need an options data source this
project doesn't have, not just another function.

All three functions return a single market-wide time series (not one per
asset), matching what ``dislocation_indicator`` expects.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def realized_volatility_indicator(prices: pd.DataFrame, lookback: int) -> pd.Series:
    """Market-wide realized volatility: each asset's own rolling return
    volatility, averaged across the universe at every bar."""
    returns = prices.pct_change()
    per_asset_vol = returns.rolling(lookback).std()
    return per_asset_vol.mean(axis=1)


def return_dispersion_indicator(prices: pd.DataFrame, lookback: int | None = None) -> pd.Series:
    """Cross-sectional dispersion of returns: the std, across assets, of
    that bar's return.

    ``lookback``, if given, smooths this with a trailing rolling average
    instead of using each bar's raw instantaneous value.
    """
    returns = prices.pct_change()
    dispersion = returns.std(axis=1)
    if lookback is not None:
        dispersion = dispersion.rolling(lookback).mean()
    return dispersion


def average_pairwise_correlation_indicator(prices: pd.DataFrame, lookback: int) -> pd.Series:
    """Average pairwise correlation of trailing ``lookback``-bar returns,
    across every asset pair, at each bar.

    Computed with an explicit per-bar loop (correlate the trailing window,
    average the off-diagonal entries) rather than a vectorized rolling-corr
    — pandas' ``.rolling().corr()`` only correlates two specific series, not
    every pair in a panel at once.
    """
    returns = prices.pct_change()
    if returns.shape[1] < 2:
        raise ValueError("need at least 2 assets to compute pairwise correlation")

    values = np.full(len(returns), np.nan)
    with warnings.catch_warnings():
        # During warmup (e.g. a window still containing the all-NaN first
        # return row), every off-diagonal entry is NaN too and nanmean's
        # "Mean of empty slice" warning is just numpy confirming that
        # expected NaN-in, NaN-out result -- not a real problem.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for t in range(lookback - 1, len(returns)):
            window = returns.iloc[t - lookback + 1 : t + 1]
            corr = window.corr().to_numpy(copy=True)
            np.fill_diagonal(corr, np.nan)
            values[t] = np.nanmean(corr)

    return pd.Series(values, index=returns.index)
