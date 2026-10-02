# quant-class-project

Statistical arbitrage research in cryptocurrencies (momentum & reversal
strategies) — Wall Street Quants course project.

- Course brief: [`ref/ClassProject.docx`](ref/ClassProject.docx)
- Specs (requirement / design / implementation / technical / feature list):
  [`specs/`](specs/)

## Status

Phase 1 (Data Infrastructure) — done: `ccxt`-based exchange OHLCV loader,
local parquet cache with freshness checks, and data validation (duplicate
removal, gap detection, insufficient-history flagging)
(`src/quant_project/data.py`). Rate limiting relies on `ccxt`'s own
`enableRateLimit`, not a hand-rolled limiter. `main.py` and the notebook
now run on real Binance data for a 15-coin universe (`build_close_price_panel`),
not synthetic data — see `specs/methodology-fixes-scope.md` item A.

Phase 2 (Signal Research) — done: momentum signals
(`src/quant_project/signals/momentum.py`) and reversal signals
(`src/quant_project/signals/reversal.py`).

Phase 3 (Backtesting & Cost Modeling) — done: unconstrained backtest engine
and execution cost model (`src/quant_project/backtest.py`,
`src/quant_project/costs.py`) and strategy weighting/combination
(`src/quant_project/combination.py` — equal-weight, inverse-volatility,
and IC-weighted combination of multiple strategy signals).

Phase 4 (Performance Reporting) — done: performance metrics and historical
view (`src/quant_project/performance.py` — cumulative gross/net returns,
annualized return/volatility, Sharpe ratio (arithmetic-mean-based, per
`specs/methodology-fixes-scope.md` C1), drawdown, and alpha/beta vs. a
benchmark, plus alpha's t-statistic and strategy/benchmark correlation,
C2) and a performance summary notebook
(`notebooks/performance_summary.ipynb`, matplotlib charts + a metrics
table — needs the `notebooks` dependency group, see Setup).
Train/test split and parameter selection (`src/quant_project/model_selection.py`
— C3/C4): momentum lookback, EMA fast/slow pair, and combination method
are each chosen by training-period Sharpe only, then every strategy is
reported separately in-sample and out-of-sample, both in `main.py` and
the notebook. The notebook (D1) leads with a stated hypothesis and
economic rationale per strategy and an overall research conclusion —
none of the three signals shows a real, cost-surviving out-of-sample
edge in this universe/period — pushing the metrics table/charts to an
appendix; see `specs/methodology-fixes-scope.md` section D for the full
writeup.

Phase 5 (Data Integrity & Reproducibility, cross-cutting) — done: input
validation on run configuration (`DataRequest` in `data.py`, and now
literal multi-exchange ingestion via `load_multi_exchange_universe`), rate
limiting (`ccxt`'s `enableRateLimit`), deterministic parquet caching, and
run/parameter audit logging (`src/quant_project/audit.py`, an append-only
JSON-Lines file at `data/audit_log.jsonl`). Macro-reversal's dislocation
indicators (`signals/indicators.py`) cover realized volatility, return
dispersion, and average pairwise correlation from spot OHLCV; implied
volatility is out of scope — it needs options market data this project
doesn't ingest.
All of the above are tested in `tests/test_quant_project.py`.
See [`specs/implementation-spec.md`](specs/implementation-spec.md) for the
full checklist; boxes are ticked there as each item is finished.

Project tooling is set up: `uv` for dependency/environment management,
`ruff` for linting, `pytest` + `pytest-cov` for testing and coverage.

## Setup

```bash
uv sync            # creates .venv/ and installs dependencies
uv sync --group notebooks   # also installs jupyter/matplotlib for notebooks/
```

## Common commands

```bash
uv run pytest      # run tests with coverage
uv run ruff check . # lint
uv run python main.py  # real data -> train/test split + param selection -> backtest -> combination -> report
uv run python optimize.py  # wider walk-forward parameter search (5 folds) -- see its own docstring
```

`optimize.py` is a separate, exploratory follow-up to `main.py`: it widens
the lookback/EMA/combination-method search and re-selects across 5
sequential walk-forward folds instead of one static split, reporting the
concatenated out-of-sample result as the honest number. As of the last
run, the answer is still "no cost-surviving edge" — if anything, the
walk-forward view makes the finding *stronger*: 1-day-ish reversal is
significantly negative in every single fold, not just one split, while
momentum/EMA crossover flip sign fold to fold (unstable, consistent with
chasing noise rather than a real effect). Widening the search further
doesn't bypass that; see the script's own docstring for the
multiple-comparisons caveat.

## Notebooks

`notebooks/performance_summary.ipynb` runs the signals → backtest →
combination → performance-report pipeline and charts the results
(cumulative returns, drawdown, and a metrics table). It needs the
`notebooks` dependency group (`uv sync --group notebooks`, see Setup).

To open and run it interactively:

```bash
uv run --group notebooks jupyter lab notebooks/performance_summary.ipynb
```

This starts Jupyter using the project's venv, so `quant_project` and its
dependencies are already importable — no separate kernel setup needed.
Once it's open, **Kernel → Restart & Run All** re-runs every cell from
scratch.

To re-run it headlessly and refresh its saved output (e.g. after changing
signal, backtest, or combination code) without opening a browser:

```bash
uv run --group notebooks jupyter nbconvert --to notebook --execute --inplace \
  notebooks/performance_summary.ipynb
```

The notebook loads the same real Binance data `main.py` uses (a 15-coin
daily-OHLCV universe since 2023-01-01, cached locally after the first
fetch) — running either one for the first time needs network access; both
reuse the cache afterward.

## Project layout

```
src/quant_project/   # package source
tests/               # tests, mirroring src/quant_project/
notebooks/           # performance summary notebook (needs `--group notebooks`)
data/raw/            # cached OHLCV parquet files (git-ignored)
data/audit_log.jsonl # run/parameter audit trail (git-ignored)
specs/               # requirement/design/implementation/technical specs
ref/                 # original course brief
```
