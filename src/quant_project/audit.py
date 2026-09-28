"""Run/parameter audit logging — Feature 7.4 in specs/feature-list.md.

Per specs/design-spec.md's "Data Integrity & Reproducibility Measures":
"Run/parameter logging for every backtest, giving an audit trail of what
configuration produced which result." This is deliberately the simplest
thing that satisfies that — an append-only JSON-Lines file, no database —
matching the local-filesystem infra the rest of the data layer
(data.py's parquet cache) already uses. Any run worth auditing (a data
load, a backtest, a combination run) calls ``log_run`` with its own
parameters/result dict; this module doesn't know or care which.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_AUDIT_LOG_PATH = Path("data/audit_log.jsonl")


def log_run(
    event_type: str,
    parameters: dict[str, Any],
    result_summary: dict[str, Any] | None = None,
    log_path: Path = DEFAULT_AUDIT_LOG_PATH,
) -> dict[str, Any]:
    """Append one audit record for a run and return the record written.

    ``parameters`` should capture exactly what configuration produced the
    run (symbols, date range, weighting method, ...); ``result_summary`` a
    small dict of what came out of it (rows loaded, cumulative return, ...).
    Values that aren't natively JSON-serializable (e.g. ``pd.Timestamp``)
    are stringified rather than raising.
    """
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event_type": event_type,
        "parameters": parameters,
        "result_summary": result_summary or {},
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


def read_audit_log(log_path: Path = DEFAULT_AUDIT_LOG_PATH) -> list[dict[str, Any]]:
    """Read back every record in the audit log, in the order they were written."""
    if not log_path.exists():
        return []
    with log_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
