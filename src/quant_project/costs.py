"""Execution cost model — Feature 4.5 in specs/feature-list.md.

Kept separate from backtest.py the way the WSQ course splits it (its
"Tcosts" lecture is separate from "PythonUnconBacktest"/"Turnover").

Figures per ref/ClassProject.docx "Execution / Slippage":
- Commissions: ~7 bps on all crypto trades.
- Slippage: assumed ~13 bps additional for market orders (total unknown /
  volume dependent, but the brief tells us to assume this).
- Market orders: 7 + 13 = 20 bps all-in.
- Limit orders: 7 bps only (no assumed slippage).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

COMMISSION_BPS: float = 7.0
MARKET_SLIPPAGE_BPS: float = 13.0
MARKET_ORDER_COST_BPS: float = COMMISSION_BPS + MARKET_SLIPPAGE_BPS  # 20 bps
LIMIT_ORDER_COST_BPS: float = COMMISSION_BPS  # 7 bps

OrderType = Literal["market", "limit"]


def cost_bps_for(order_type: OrderType) -> float:
    """All-in execution cost in basis points for the given order type."""
    if order_type == "market":
        return MARKET_ORDER_COST_BPS
    if order_type == "limit":
        return LIMIT_ORDER_COST_BPS
    raise ValueError(f"Unknown order_type: {order_type!r} (expected 'market' or 'limit')")


def apply_transaction_costs(
    turnover: pd.Series,
    order_type: OrderType = "market",
) -> pd.Series:
    """Convert per-period portfolio turnover into a return drag.

    ``turnover`` is expected as the fraction of gross portfolio value traded
    in each period (e.g. 0.5 means 50% of the book was turned over that
    period). Returns a series of the same index with the (positive) cost to
    subtract from strategy returns.
    """
    cost_rate = cost_bps_for(order_type) / 10_000.0
    return turnover * cost_rate


def net_of_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    order_type: OrderType = "market",
) -> pd.Series:
    """Strategy returns after subtracting estimated execution costs."""
    return gross_returns - apply_transaction_costs(turnover, order_type=order_type)
