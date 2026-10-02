"""Train/test split and candidate selection — C3/C4 in
specs/methodology-fixes-scope.md.

Three small, independent primitives:

- ``split_prices`` (C3): a strictly chronological, non-overlapping
  train/test split of a price panel — never shuffled, since shuffling a
  time series for a train/test split would itself be a look-ahead bug (a
  "training" bar could land after a "test" bar it's supposed to be
  evaluated against).
- ``select_best`` (C3): picks whichever named candidate has the highest
  score. It doesn't know or care what a "candidate" or a "score" is —
  callers compute each candidate's selection metric (e.g.
  ``performance.sharpe_ratio`` of a signal's net returns, *restricted to
  the training period*) however fits their pipeline, then hand in a
  ``{name: score}`` dict. Keeping this generic means the same function
  selects a momentum lookback, an EMA fast/slow pair, *and* a combination
  method — anything reducible to "which of these named options scored best
  on training data" — without this module needing to know anything about
  signals, backtests, or combination methods.
- ``slice_backtest_result`` (C4): restricts every field of an already-run
  ``BacktestResult`` to a given date index, so the *same* full-history
  backtest can be reported separately in-sample (training dates) and
  out-of-sample (test dates) without re-running anything.

None of these run a backtest or touch a strategy's parameters themselves —
wire them into a concrete pipeline (see ``main.py``) by: (1) splitting the
price panel, (2) backtesting each candidate over the *full* price history
(so any path-dependent signal's internal state carries over naturally
across the split boundary, exactly as it would in a live walk-forward run
— only the dates used for *scoring* candidates are restricted to the
training period, never the signal computation itself), (3) scoring each
candidate's result sliced to training dates only (``slice_backtest_result``
+ ``performance.sharpe_ratio``, typically), (4) calling ``select_best``,
then (5) reporting the winner's performance separately over the training
and test date ranges (``slice_backtest_result`` again).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_project.backtest import BacktestResult


@dataclass
class TrainTestSplit:
    train_prices: pd.DataFrame
    test_prices: pd.DataFrame
    split_date: pd.Timestamp
    """The first bar of the test period — also the first bar *not* in
    ``train_prices``."""


def split_prices(prices: pd.DataFrame, train_fraction: float = 0.7) -> TrainTestSplit:
    """A strictly chronological, non-overlapping train/test split.

    The first ``train_fraction`` of bars (by position, not by calendar
    span) become ``train_prices``; everything from ``split_date`` onward
    becomes ``test_prices``. Never shuffled — shuffling would let a
    "training" bar land after the "test" bar it's meant to be evaluated
    against, which is itself a look-ahead bug.

    ``train_fraction`` defaults to 0.7 (a plain, commonly-used 70/30
    split) — the mentor feedback that scoped this didn't specify a ratio,
    and nothing about this project's data favors a different one.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if len(prices) < 2:
        raise ValueError("need at least 2 bars to split")

    split_idx = round(len(prices) * train_fraction)
    split_idx = min(max(split_idx, 1), len(prices) - 1)
    split_date = prices.index[split_idx]
    return TrainTestSplit(
        train_prices=prices.iloc[:split_idx],
        test_prices=prices.iloc[split_idx:],
        split_date=split_date,
    )


def select_best(candidate_scores: dict[str, float]) -> str:
    """Return the name of whichever candidate has the highest score.

    ``candidate_scores`` is typically each candidate's Sharpe ratio (or
    similar) computed *only* over a training period — this function has no
    notion of time at all, so keeping it that way is entirely the caller's
    responsibility (see the module docstring).

    Raises if ``candidate_scores`` is empty or every score is NaN (e.g.
    every candidate was still in its warm-up period for the whole training
    window).
    """
    valid = {name: score for name, score in candidate_scores.items() if pd.notna(score)}
    if not valid:
        raise ValueError("no candidate has a usable (non-NaN) score")
    return max(valid, key=lambda name: valid[name])


def slice_backtest_result(result: BacktestResult, index: pd.Index) -> BacktestResult:
    """Restrict every field of a ``BacktestResult`` to ``index`` (e.g. a
    ``TrainTestSplit``'s ``train_prices.index`` or ``test_prices.index``),
    so one full-history backtest run can be reported in-sample and
    out-of-sample separately without re-running it twice."""
    return BacktestResult(
        weights=result.weights.reindex(index),
        turnover=result.turnover.reindex(index),
        gross_returns=result.gross_returns.reindex(index),
        net_returns=result.net_returns.reindex(index),
    )
