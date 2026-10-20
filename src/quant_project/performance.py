"""Performance reporting — Feature 6 in specs/feature-list.md.

Turns a ``backtest.BacktestResult`` (or any per-period return series) into
the standard set of performance figures a researcher reviews after a run:
cumulative returns, annualized return/volatility, Sharpe ratio, max
drawdown, and alpha/beta (plus significance and correlation) vs. a
benchmark.

Alpha/beta (6.5) is computed as closed-form OLS (beta = cov/var, alpha from
the sample means) rather than via ``scipy``/``statsmodels`` regression —
those aren't otherwise project dependencies, and a single-regressor-plus-
intercept fit needs nothing more than covariance and variance (and, for
the alpha t-statistic added per specs/methodology-fixes-scope.md C2, the
standard closed-form OLS intercept standard-error formula).

``periods_per_year`` defaults to 365 throughout, since the underlying
strategies trade crypto (24/7, no market closures) rather than a
5-day-a-week equity calendar; pass a different value for a coarser
(e.g. hourly) timeframe.

The last section asks how much of a Sharpe ratio could be luck: a
bootstrap confidence interval for one strategy, and the probabilistic and
deflated Sharpe ratios (Bailey & López de Prado) for the best of many
strategies tried. The normal distribution comes from the standard
library's ``statistics.NormalDist``, for the same reason there's no
``scipy`` above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from quant_project.backtest import BacktestResult

DEFAULT_PERIODS_PER_YEAR = 365

EULER_MASCHERONI = 0.5772156649015329


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
    """Annualized Sharpe ratio: ``mean(returns) / std(returns) * sqrt(periods_per_year)``.

    This is the standard arithmetic-mean-based Sharpe ratio (per
    specs/methodology-fixes-scope.md C1) — *not*
    ``annualized_return(returns) / annualized_volatility(returns)``, which
    this function computed before. That alternative divides a
    *geometrically compounded* return by vol; compounding is nonlinear, so
    it over- or under-states Sharpe relative to the conventional
    arithmetic-mean definition depending on the sign and size of the
    returns involved, rather than just rescaling it.

    ``risk_free_rate`` is annualized; it's de-annualized by simple division
    (``risk_free_rate / periods_per_year``) before being subtracted from the
    per-period mean, so the subtraction happens in the same per-period units
    as everything else in this formula. Default 0.0 (a standard
    simplification absent a stated benchmark rate).
    """
    std = returns.std()
    if std == 0.0 or np.isnan(std):
        return float("nan")
    per_period_risk_free = risk_free_rate / periods_per_year
    excess_return = returns.mean() - per_period_risk_free
    return (excess_return / std) * np.sqrt(periods_per_year)


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
# 6.5 Alpha and beta vs. a benchmark, plus alpha's significance (C2)
# ---------------------------------------------------------------------------
@dataclass
class AlphaBetaResult:
    alpha: float
    """Annualized alpha, compounded from the per-period OLS intercept."""
    beta: float
    alpha_t_stat: float
    """t-statistic of the *per-period* OLS intercept, against the null
    hypothesis that the true per-period alpha is zero. Computed on the
    per-period (not annualized) alpha deliberately: annualizing is a
    nonlinear compounding transform, so there's no such thing as an
    "annualized t-stat" — this tests whether the per-period estimate
    itself is distinguishable from zero given its own standard error."""
    correlation: float
    """Pearson correlation between strategy and benchmark returns — a
    plain diagnostic of how closely the two move together, distinct from
    beta (which is scaled by their relative variances)."""


def alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> AlphaBetaResult:
    """OLS alpha/beta of ``returns`` vs. ``benchmark_returns``, with alpha's
    t-statistic and the strategy/benchmark correlation.

    beta = Cov(strategy, benchmark) / Var(benchmark); alpha is the
    per-period intercept (``mean(strategy) - beta * mean(benchmark)``),
    compounded up to an annualized figure. ``alpha_t_stat`` uses the
    standard closed-form OLS intercept standard error,
    ``se(alpha) = sigma_hat * sqrt(1/n + mean(benchmark)^2 / Sxx)``, where
    ``sigma_hat`` is the residual standard error and
    ``Sxx = sum((benchmark - mean(benchmark))^2)``. Only periods where both
    series have a value are used; at least 3 are required (2 parameters are
    fit, leaving 1 residual degree of freedom at the minimum).
    """
    aligned = pd.concat(
        [returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1
    ).dropna()
    n = len(aligned)
    if n < 3:
        raise ValueError(
            "need at least 3 overlapping periods to estimate alpha/beta and its t-stat"
        )

    strategy, benchmark = aligned["strategy"], aligned["benchmark"]
    benchmark_mean = benchmark.mean()
    benchmark_dispersion = ((benchmark - benchmark_mean) ** 2).sum()
    if benchmark_dispersion == 0.0:
        raise ValueError("benchmark has zero variance; cannot estimate beta")

    beta = strategy.cov(benchmark) / (benchmark_dispersion / (n - 1))
    alpha_per_period = strategy.mean() - beta * benchmark_mean
    alpha_annualized = (1 + alpha_per_period) ** periods_per_year - 1.0

    residuals = strategy - (alpha_per_period + beta * benchmark)
    residual_variance = (residuals**2).sum() / (n - 2)
    alpha_se = np.sqrt(residual_variance * (1.0 / n + benchmark_mean**2 / benchmark_dispersion))
    alpha_t_stat = alpha_per_period / alpha_se if alpha_se > 0.0 else float("nan")

    correlation = strategy.corr(benchmark)

    return AlphaBetaResult(
        alpha=alpha_annualized,
        beta=beta,
        alpha_t_stat=alpha_t_stat,
        correlation=correlation,
    )


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
    alpha_t_stat: float | None
    """t-statistic of the per-period alpha (see ``AlphaBetaResult``), or
    None if no benchmark was given."""
    correlation: float | None
    """Pearson correlation between strategy and benchmark returns, or None
    if no benchmark was given."""


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
    alpha_t_stat: float | None = None
    correlation: float | None = None
    if benchmark_returns is not None:
        alpha_beta_result = alpha_beta(net_returns, benchmark_returns, periods_per_year)
        alpha = alpha_beta_result.alpha
        beta = alpha_beta_result.beta
        alpha_t_stat = alpha_beta_result.alpha_t_stat
        correlation = alpha_beta_result.correlation

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
        alpha_t_stat=alpha_t_stat,
        correlation=correlation,
    )


# ---------------------------------------------------------------------------
# How much of a Sharpe ratio could be luck?
# ---------------------------------------------------------------------------
def sharpe_confidence_interval(
    returns: pd.Series,
    level: float = 0.95,
    block_length: int = 10,
    n_resamples: int = 5000,
    seed: int = 0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> tuple[float, float]:
    """Moving-block bootstrap confidence interval for the annualized Sharpe ratio.

    Resamples the return series in overlapping blocks of ``block_length``
    periods (10 days by default, about two weeks), recomputes the Sharpe
    ratio on each resample, and returns the percentile interval. Blocks
    keep any short-run autocorrelation; resampling, rather than a formula,
    keeps the fat tails. That matters here: the usual analytic standard
    error adjusts for skew in a way that *narrows* the interval when one
    huge positive day dominates a series, which is backwards for a result
    that depends on that one day. ``seed`` makes the interval reproducible.
    """
    values = returns.dropna().to_numpy()
    n = len(values)
    if n < 2 * block_length:
        raise ValueError(f"need at least {2 * block_length} returns for a block bootstrap")

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block_length)
    starts = rng.integers(0, n - block_length + 1, size=(n_resamples, n_blocks))
    index = (starts[:, :, None] + np.arange(block_length)).reshape(n_resamples, -1)[:, :n]
    resampled = values[index]
    sharpes = resampled.mean(axis=1) / resampled.std(axis=1, ddof=1) * np.sqrt(periods_per_year)

    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.nanpercentile(sharpes, [tail, 100.0 - tail])
    return float(low), float(high)


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    benchmark_sharpe: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Probability that the true Sharpe ratio beats ``benchmark_sharpe``
    (annualized), given this return series (Bailey & López de Prado, 2012).

    Uses the standard error of a Sharpe estimate allowing for skewness and
    fat tails: per period, ``sqrt((1 - skew*SR + (kurtosis - 1)/4 * SR^2) / (T - 1))``,
    with ``kurtosis`` the ordinary (non-excess) kurtosis.
    """
    values = returns.dropna()
    n = len(values)
    sharpe = values.mean() / values.std()
    benchmark = benchmark_sharpe / np.sqrt(periods_per_year)
    skew = values.skew()
    kurtosis = values.kurt() + 3.0
    variance_factor = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    z = (sharpe - benchmark) * np.sqrt(n - 1) / np.sqrt(variance_factor)
    return NormalDist().cdf(float(z))


def expected_maximum_sharpe(
    n_trials: float,
    n_observations: int,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """The annualized Sharpe ratio the best of ``n_trials`` independent
    strategies with *no* edge would show, on average, by luck alone.

    The approximation from Bailey & López de Prado's "False Strategy
    Theorem": ``sd * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)))``,
    with ``g`` the Euler-Mascheroni constant. For ``sd`` (the spread of
    Sharpe estimates across trials) this uses what the no-edge hypothesis
    itself implies, ``1 / sqrt(T - 1)`` per period, rather than the spread
    of the trials actually run. The trials' own spread mixes luck with real
    differences (a strategy that pays heavy costs really is worse), and
    would set the bar for luck too high. One trial needs no deflating, so
    it returns 0.
    """
    if n_trials <= 1:
        return 0.0
    normal = NormalDist()
    expected_max_z = (1.0 - EULER_MASCHERONI) * normal.inv_cdf(
        1.0 - 1.0 / n_trials
    ) + EULER_MASCHERONI * normal.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    per_period_sd = 1.0 / np.sqrt(n_observations - 1)
    return float(per_period_sd * expected_max_z * np.sqrt(periods_per_year))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: float,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Probability that a strategy's true Sharpe ratio beats what the best of
    ``n_trials`` no-edge strategies would show by luck (Bailey & López de
    Prado, 2014): the probabilistic Sharpe ratio against
    ``expected_maximum_sharpe`` instead of against zero. Use it on the
    *best* of several strategies tried; 0.95 is the usual bar.
    """
    n_observations = len(returns.dropna())
    hurdle = expected_maximum_sharpe(n_trials, n_observations, periods_per_year)
    return probabilistic_sharpe_ratio(returns, hurdle, periods_per_year)


def effective_number_of_trials(trial_returns: pd.DataFrame) -> float:
    """How many *independent* strategies a set of correlated ones amounts to.

    ``N^2 / sum of squared pairwise correlations`` (the participation ratio
    of the correlation matrix's eigenvalues): N for uncorrelated trials, 1
    when every trial is the same bet. Squaring means a strategy and its
    exact opposite (momentum and reversal over the same lookback) count as
    one bet, which is right: trying both directions is one idea. It's a
    rough estimate; López de Prado's later work clusters the trials instead.
    """
    correlations = trial_returns.corr().to_numpy()
    n = correlations.shape[0]
    return float(n**2 / np.nansum(correlations**2))
