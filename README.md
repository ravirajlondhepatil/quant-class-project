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
`enableRateLimit`, not a hand-rolled limiter.

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
annualized return/volatility, Sharpe ratio, drawdown, and alpha/beta vs. a
benchmark) and a performance summary notebook
(`notebooks/performance_summary.ipynb`, matplotlib charts + a metrics
table — needs the `notebooks` dependency group, see Setup).

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
uv run python main.py  # manual smoke-run: synthetic prices -> signals -> backtest -> combination
```

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

The notebook runs on the same small synthetic price panel `main.py` uses
(Phase 1's real exchange loader needs network access neither this notebook
nor `main.py` requires) — swap in `quant_project.data.load_universe_ohlcv`
once you're ready to point it at a real exchange.

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
