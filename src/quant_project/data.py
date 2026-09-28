"""Data infrastructure — Phase 1 in specs/implementation-spec.md, plus the
data-layer half of Feature 7 (Data Integrity & Reproducibility) in
specs/feature-list.md.

Four pieces, in the order a request flows through them:

- ``DataRequest`` (1.2) validates the run configuration up front — exchange,
  symbol universe, timeframe, date range (7.1: input validation/sanitization
  on run configuration).
- ``build_exchange_client`` / ``fetch_ohlcv_from_exchange`` (1.1) pull OHLCV
  from an exchange via ``ccxt``, paginating through ``fetch_ohlcv`` calls.
  Rate limiting (7.2) is *not* hand-rolled here — ``build_exchange_client``
  just turns on ``ccxt``'s own ``enableRateLimit``, which makes every
  exchange client sleep between calls per that exchange's documented limit;
  re-implementing that would just be a worse copy of what ``ccxt`` already
  does correctly per-exchange.
- ``cache_path`` / ``read_cache`` / ``write_cache`` / ``is_cache_fresh``
  (1.3, 1.4) are a local parquet cache, keyed purely by
  (exchange, symbol, timeframe) — the same request always resolves to the
  same file, so results are reproducible run to run (7.3) without any
  hidden state beyond "has this been fetched before".
- ``validate_ohlcv`` (1.5) removes duplicate timestamps, detects gaps against
  the timeframe's expected frequency, and flags insufficient history.

``load_symbol_ohlcv`` / ``load_universe_ohlcv`` wire all four together and
are what most callers should use; the pieces above are exposed separately
because each is independently useful and independently tested.
``load_multi_exchange_universe`` is the same thing across several
exchanges at once — literal "multi-exchange" ingestion (1.1) — merging one
``DataRequest`` per exchange into a single symbol -> data mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_project.audit import DEFAULT_AUDIT_LOG_PATH, log_run

DEFAULT_CACHE_DIR = Path("data/raw")

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# ccxt's unified timeframe strings -> the pandas frequency alias used to
# build the "expected" index for gap detection in `validate_ohlcv`.
SUPPORTED_TIMEFRAMES: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1w": "1W",
}


# ---------------------------------------------------------------------------
# 1.2 / 7.1 Symbol universe & timeframe configuration, validated up front
# ---------------------------------------------------------------------------
@dataclass
class DataRequest:
    """A validated request for OHLCV data: one exchange, a symbol universe,
    one timeframe, and a date range.

    Raises ``ValueError`` at construction time (7.1) if the exchange id or
    any symbol is blank, the timeframe isn't one ``validate_ohlcv`` knows how
    to gap-check, or the date range is empty/inverted — so a bad
    configuration fails before any network call is made, not partway
    through a fetch.
    """

    exchange_id: str
    symbols: list[str]
    timeframe: str
    start: str | pd.Timestamp
    end: str | pd.Timestamp

    def __post_init__(self) -> None:
        if not self.exchange_id or not self.exchange_id.strip():
            raise ValueError("exchange_id must be a non-empty string")
        if not self.symbols:
            raise ValueError("symbols must be a non-empty list")
        for symbol in self.symbols:
            if not symbol or not symbol.strip():
                raise ValueError(f"invalid symbol: {symbol!r}")
        if self.timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported timeframe: {self.timeframe!r} "
                f"(expected one of {sorted(SUPPORTED_TIMEFRAMES)})"
            )
        self.start = pd.Timestamp(self.start)
        self.end = pd.Timestamp(self.end)
        if self.start >= self.end:
            raise ValueError(f"start ({self.start}) must be before end ({self.end})")


# ---------------------------------------------------------------------------
# 1.1 / 7.2 Exchange OHLCV retrieval
# ---------------------------------------------------------------------------
def build_exchange_client(exchange_id: str) -> Any:
    """Construct a rate-limited ``ccxt`` exchange client by id (e.g. "binance").

    ``enableRateLimit=True`` is what satisfies 7.2 — ccxt then sleeps
    between calls per that exchange's own documented rate limit.
    """
    import ccxt

    try:
        exchange_class = getattr(ccxt, exchange_id)
    except AttributeError as exc:
        raise ValueError(f"Unknown ccxt exchange id: {exchange_id!r}") from exc
    return exchange_class({"enableRateLimit": True})


def fetch_ohlcv_from_exchange(
    exchange: Any,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch OHLCV for one symbol over [start, end], paginating as needed.

    ``exchange`` is any object exposing ccxt's ``fetch_ohlcv(symbol,
    timeframe, since, limit)`` -> list of ``[ts_ms, open, high, low, close,
    volume]`` rows — a real ``ccxt`` client, or a test double. Pagination
    stops once a page starts at or before its own cursor (the exchange
    isn't advancing) or returns fewer than ``limit`` rows (no more data).
    """
    since_ms = int(pd.Timestamp(start).timestamp() * 1000)
    until_ms = int(pd.Timestamp(end).timestamp() * 1000)

    rows: list[list] = []
    cursor = since_ms
    while cursor <= until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = batch[-1][0] + 1
        if next_cursor <= cursor or len(batch) < limit:
            break
        cursor = next_cursor

    if not rows:
        empty_index = pd.DatetimeIndex([], name="timestamp")
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=empty_index)

    frame = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms")
    frame = frame.set_index("timestamp").sort_index()
    return frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]


# ---------------------------------------------------------------------------
# 1.3 / 1.4 / 7.3 Local parquet cache, keyed deterministically by request
# ---------------------------------------------------------------------------
def cache_path(
    exchange_id: str, symbol: str, timeframe: str, cache_dir: Path = DEFAULT_CACHE_DIR
) -> Path:
    """Deterministic cache file location for one (exchange, symbol, timeframe)."""
    safe_symbol = symbol.replace("/", "-")
    return cache_dir / exchange_id / timeframe / f"{safe_symbol}.parquet"


def read_cache(path: Path) -> pd.DataFrame | None:
    """Load a cached OHLCV frame, or None if nothing is cached yet."""
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_cache(df: pd.DataFrame, path: Path) -> None:
    """Persist an OHLCV frame to the cache, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def is_cache_fresh(cached: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Whether a cached frame already fully covers [start, end] — if so,
    it can be reused instead of re-fetching (1.4)."""
    if cached is None or cached.empty:
        return False
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    return bool(cached.index.min() <= start and cached.index.max() >= end)


# ---------------------------------------------------------------------------
# 1.5 Data validation
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    cleaned: pd.DataFrame
    """The input with duplicate timestamps dropped, sorted by time."""
    duplicates_removed: int
    gap_timestamps: pd.DatetimeIndex
    """Expected bars (per the timeframe's frequency) missing from ``cleaned``."""
    has_sufficient_history: bool


def validate_ohlcv(df: pd.DataFrame, timeframe: str, min_periods: int = 30) -> ValidationReport:
    """Duplicate-timestamp removal, gap detection, and a history-length flag.

    Gaps are found by comparing ``cleaned``'s index against the full
    expected index (per the timeframe's frequency) spanning its own
    min-to-max range — so a symbol that's simply short (fewer than
    ``min_periods`` bars) is reported via ``has_sufficient_history``, not as
    one giant "gap" before its first real bar.
    """
    duplicates_removed = int(df.index.duplicated().sum())
    cleaned = df[~df.index.duplicated(keep="first")].sort_index()

    if len(cleaned) >= 2:
        freq = SUPPORTED_TIMEFRAMES[timeframe]
        expected_index = pd.date_range(cleaned.index.min(), cleaned.index.max(), freq=freq)
        gap_timestamps = expected_index.difference(cleaned.index)
    else:
        gap_timestamps = pd.DatetimeIndex([])

    return ValidationReport(
        cleaned=cleaned,
        duplicates_removed=duplicates_removed,
        gap_timestamps=gap_timestamps,
        has_sufficient_history=len(cleaned) >= min_periods,
    )


# ---------------------------------------------------------------------------
# Orchestration: request -> (cache or fetch) -> validate -> audit log
# ---------------------------------------------------------------------------
def load_symbol_ohlcv(
    request: DataRequest,
    symbol: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    exchange: Any = None,
    force_refresh: bool = False,
    audit_log_path: Path | None = DEFAULT_AUDIT_LOG_PATH,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Load, validate, and cache OHLCV for one symbol in ``request``.

    Reuses the cache when it already covers ``[request.start, request.end]``
    (1.4); otherwise fetches from the exchange (building a rate-limited
    client via ``build_exchange_client`` if none was supplied), merges with
    whatever was already cached, and rewrites the cache. Every call is
    audit-logged (7.4) unless ``audit_log_path`` is None.
    """
    path = cache_path(request.exchange_id, symbol, request.timeframe, cache_dir)
    cached = None if force_refresh else read_cache(path)

    if cached is not None and is_cache_fresh(cached, request.start, request.end):
        raw, source = cached, "cache"
    else:
        if exchange is None:
            exchange = build_exchange_client(request.exchange_id)
        fetched = fetch_ohlcv_from_exchange(
            exchange, symbol, request.timeframe, request.start, request.end
        )
        raw = fetched if cached is None else pd.concat([cached, fetched])
        write_cache(raw, path)
        source = "fetched"

    report = validate_ohlcv(raw, request.timeframe)
    windowed = report.cleaned.loc[request.start : request.end]

    if audit_log_path is not None:
        log_run(
            "data_load",
            parameters={
                "exchange_id": request.exchange_id,
                "symbol": symbol,
                "timeframe": request.timeframe,
                "start": request.start,
                "end": request.end,
                "force_refresh": force_refresh,
            },
            result_summary={
                "source": source,
                "rows": len(windowed),
                "duplicates_removed": report.duplicates_removed,
                "gap_count": len(report.gap_timestamps),
                "sufficient_history": report.has_sufficient_history,
            },
            log_path=audit_log_path,
        )
    return windowed, report


def load_universe_ohlcv(
    request: DataRequest, **kwargs
) -> dict[str, tuple[pd.DataFrame, ValidationReport]]:
    """``load_symbol_ohlcv`` for every symbol in ``request.symbols``."""
    return {symbol: load_symbol_ohlcv(request, symbol, **kwargs) for symbol in request.symbols}


def load_multi_exchange_universe(
    requests: list[DataRequest],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    exchanges: dict[str, Any] | None = None,
    force_refresh: bool = False,
    audit_log_path: Path | None = DEFAULT_AUDIT_LOG_PATH,
) -> dict[str, tuple[pd.DataFrame, ValidationReport]]:
    """1.1's "multi-exchange" ingestion: load a symbol universe spread
    across several exchanges in one call.

    Each ``DataRequest`` in ``requests`` supplies its own exchange id,
    symbols, timeframe, and date range (e.g. one request for Binance
    symbols, another for Kraken symbols); results are merged into one dict
    keyed by symbol. ``exchanges`` optionally maps an exchange id to an
    already-built client (real or a test double) to use for that request's
    fetches, instead of each request building its own via
    ``build_exchange_client``.

    Raises ``ValueError`` if the same symbol is requested from more than
    one exchange — which exchange's data should win is ambiguous, so this
    doesn't silently pick one.
    """
    exchanges = exchanges or {}
    combined: dict[str, tuple[pd.DataFrame, ValidationReport]] = {}
    for request in requests:
        for symbol, result in load_universe_ohlcv(
            request,
            cache_dir=cache_dir,
            exchange=exchanges.get(request.exchange_id),
            force_refresh=force_refresh,
            audit_log_path=audit_log_path,
        ).items():
            if symbol in combined:
                raise ValueError(f"symbol {symbol!r} requested from more than one exchange")
            combined[symbol] = result
    return combined
