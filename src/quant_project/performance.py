"""Performance reporting — Feature 6 in specs/feature-list.md.

Turns a ``backtest.BacktestResult`` (or any per-period return series) into
the standard set of performance figures a researcher reviews after a run:
cumulative returns, annualized return/volatility, Sharpe ratio, max
drawdown, and alpha/beta vs. a benchmark.

Alpha/beta (6.5) is computed as closed-form OLS (beta = cov/var, alpha from
the sample means) rather than via ``scipy``/``statsmodels`` regression —
those aren't otherwise project dependencies, and a single-regressor-plus-
intercept fit needs nothing more than covariance and variance.

``periods_per_year`` defaults to 365 throughout, since the underlying
strategies trade crypto (24/7, no market closures) rather than a
5-day-a-week equity calendar; pass a different value for a coarser
(e.g. hourly) timeframe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_project.backtest import BacktestResult

DEFAULT_PERIODS_PER_YEAR = 365


# ---------------------------------------------------------------------------
# 6.1 Gross vs. net-of-cost cumulative returns
# ---------------------------------------------------------------------------
def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Compounded cumulative return series from per-period returns.

    NaN periods (no position held) are treated as a zero return for that
    period rather than breaking the compounding chain.
    """
    return (1 + returns.fillna(0.0)).cumprod() - 1.0


# ---------------------------------------------------------------------------
# 6.2 Annualized return and annualized volatility
# ---------------------------------------------------------------------------
def annualized_return(
    returns: pd.Series, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
) -> float:
    """Geometric (compounded) annualized return."""
    n_periods = len(returns)
    if n_periods == 0:
        return float("nan")
    total_growth = (1 + returns.fillna(0.0)).prod()
    return total_growth ** (periods_per_year / n_periods) - 1.0


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
) -> float:
    """Annualized standard deviation of per-period returns."""
    return returns.std() * np.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# 6.3 Sharpe ratio
# ---------------------------------------------------------------------------
def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sharpe ratio: excess annualized return over annualized vol.

    ``risk_free_rate`` is annualized; default 0.0 (a standard simplification
    absent a stated benchmark rate).
    """
    vol = annualized_volatility(returns, periods_per_year)
    if vol == 0.0 or np.isnan(vol):
        return float("nan")
    excess_return = annualized_return(returns, periods_per_year) - risk_free_rate
    return excess_return / vol


# ---------------------------------------------------------------------------
# 6.4 / 6.6 Max drawdown and its full historical path
# ---------------------------------------------------------------------------
def drawdown_series(returns: pd.Series) -> pd.Series:
    """Peak-to-date decline of the cumulative wealth curve, at every period.

    A negative fraction (0.0 = at a new high, -0.2 = 20% below the highest
    wealth reached so far). This is the "historical performance view" (6.6)
    of drawdown; ``max_drawdown`` (6.4) is just this series' minimum.
    """
    wealth = (1 + returns.fillna(0.0)).cumprod()
    running_max = wealth.cummax()
    return wealth / running_max - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline over the full return history."""
    return drawdown_series(returns).min()


# ---------------------------------------------------------------------------
# 6.5 Alpha and beta vs. a benchmark
# ---------------------------------------------------------------------------
def alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> tuple[float, float]:
    """OLS alpha (annualized) and beta of ``returns`` vs. ``benchmark_returns``.

    beta = Cov(strategy, benchmark) / Var(benchmark); alpha is the
    per-period intercept (``mean(strategy) - beta * mean(benchmark)``),
    compounded up to an annualized figure. Only periods where both series
    have a value are used.
    """
    aligned = pd.concat(
        [returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1
    ).dropna()
    if len(aligned) < 2:
        raise ValueError("need at least 2 overlapping periods to estimate alpha/beta")

    benchmark_variance = aligned["benchmark"].var()
    if benchmark_variance == 0.0:
        raise ValueError("benchmark has zero variance; cannot estimate beta")

    beta = aligned["strategy"].cov(aligned["benchmark"]) / benchmark_variance
    alpha_per_period = aligned["strategy"].mean() - beta * aligned["benchmark"].mean()
    alpha_annualized = (1 + alpha_per_period) ** periods_per_year - 1.0
    return alpha_annualized, beta


# ---------------------------------------------------------------------------
# 6.6 / 6.7 Historical performance view, regenerated fresh on every call
# ---------------------------------------------------------------------------
@dataclass
class PerformanceReport:
    cumulative_gross: pd.Series
    """Full-history gross cumulative return series."""
    cumulative_net: pd.Series
    """Full-history net-of-cost cumulative return series."""
    drawdown: pd.Series
    """Full-history net-of-cost drawdown series."""
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    alpha: float | None
    """Annualized alpha vs. the benchmark, or None if no benchmark was given."""
    beta: float | None
    """Beta vs. the benchmark, or None if no benchmark was given."""


def build_performance_report(
    result: BacktestResult,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0,
    benchmark_returns: pd.Series | None = None,
) -> PerformanceReport:
    """Assemble the full performance report (6.1-6.6) for one backtest result.

    All figures (Sharpe, drawdown, alpha/beta, ...) are computed from
    ``result.net_returns`` — the after-cost return stream — except the
    gross cumulative return series, kept alongside the net one so both are
    visible per 6.1. Since this is a plain function of whatever
    ``BacktestResult`` it's given, re-running the backtest and calling this
    again on the new result *is* the report regeneration (6.7): there is no
    separate cached/stale report to invalidate.
    """
    net_returns = result.net_returns

    alpha: float | None = None
    beta: float | None = None
    if benchmark_returns is not None:
        alpha, beta = alpha_beta(net_returns, benchmark_returns, periods_per_year)

    return PerformanceReport(
        cumulative_gross=cumulative_returns(result.gross_returns),
        cumulative_net=cumulative_returns(net_returns),
        drawdown=drawdown_series(net_returns),
        annualized_return=annualized_return(net_returns, periods_per_year),
        annualized_volatility=annualized_volatility(net_returns, periods_per_year),
        sharpe_ratio=sharpe_ratio(net_returns, periods_per_year, risk_free_rate),
        max_drawdown=max_drawdown(net_returns),
        alpha=alpha,
        beta=beta,
    )
