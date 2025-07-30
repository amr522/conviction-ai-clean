#!/usr/bin/env python3
"""
Feast entity definitions
"""
from feast import Entity, ValueType

# Stock/Option symbol entity for 36-stock training universe
ticker = Entity(
    name="ticker",
    value_type=ValueType.STRING,
    description="Stock or option symbol identifier for 36-stock universe + QQQ/SPY ETF options",
    # Training Universe: AAPL, GOOGL, META, MSFT, NFLX, SMCI, AMD, NVDA, PLTR,
    # BAC, GS, JPM, MA, MS, V, ABBV, JNJ, MRK, PFE, UNH, DIS, NKE, SBUX, WMT, AMZN,
    # CVX, XOM, BA, CAT, GE, COIN, HOOD, MSTR, TSLA, QQQ, SPY
)

# Date entity for time-based features
date_entity = Entity(
    name="date",
    value_type=ValueType.STRING,
    description="Date identifier for daily features",
)
