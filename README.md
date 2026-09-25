# quant-class-project

Statistical arbitrage research in cryptocurrencies (momentum & reversal
strategies) — Wall Street Quants course project.

- Course brief: [`ref/ClassProject.docx`](ref/ClassProject.docx)
- Specs (requirement / design / implementation / technical / feature list):
  [`specs/`](specs/)

## Status

Phase 1 (Data Infrastructure) — not started.
Phase 2 (Signal Research) — momentum signals done (`src/quant_project/signals/momentum.py`,
tested in `tests/test_momentum.py`); reversal signals not started.
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
```

## Project layout

```
src/quant_project/   # package source
tests/               # tests, mirroring src/quant_project/
data/raw/            # cached OHLCV parquet files (git-ignored)
specs/               # requirement/design/implementation/technical specs
ref/                 # original course brief
```
