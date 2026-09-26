# quant-class-project

Statistical arbitrage research in cryptocurrencies (momentum & reversal
strategies) — Wall Street Quants course project.

- Course brief: [`ref/ClassProject.docx`](ref/ClassProject.docx)
- Specs (requirement / design / implementation / technical / feature list):
  [`specs/`](specs/)

## Status

Phase 1 (Data Infrastructure) — not started.
Phase 2 (Signal Research) — done: momentum signals
(`src/quant_project/signals/momentum.py`) and reversal signals
(`src/quant_project/signals/reversal.py`).
Phase 3 (Backtesting & Cost Modeling) — done: unconstrained backtest engine
and execution cost model (`src/quant_project/backtest.py`,
`src/quant_project/costs.py`) and strategy weighting/combination
(`src/quant_project/combination.py` — equal-weight, inverse-volatility,
and IC-weighted combination of multiple strategy signals).
All of the above are tested in `tests/test_quant_project.py`.
See [`specs/implementation-spec.md`](specs/implementation-spec.md) for the
full checklist; boxes are ticked there as each item is finished.

Project tooling is set up: `uv` for dependency/environment management,
`ruff` for linting, `pytest` + `pytest-cov` for testing and coverage.

## Setup

```bash
uv sync            # creates .venv/ and installs dependencies
```

## Common commands

```bash
uv run pytest      # run tests with coverage
uv run ruff check . # lint
uv run python main.py  # manual smoke-run: synthetic prices -> signals -> backtest -> combination
```

## Project layout

```
src/quant_project/   # package source
tests/               # tests, mirroring src/quant_project/
data/raw/            # cached OHLCV parquet files (git-ignored)
specs/               # requirement/design/implementation/technical specs
ref/                 # original course brief
```
