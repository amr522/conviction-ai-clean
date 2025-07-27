#!/usr/bin/env python3
"""
Feast entity definitions
"""
from feast import Entity, ValueType

# Stock/Option symbol entity
ticker = Entity(
    name="ticker",
    value_type=ValueType.STRING,
    description="Stock or option symbol identifier"
)

# Date entity for time-based features
date_entity = Entity(
    name="date",
    value_type=ValueType.STRING,
    description="Date identifier for daily features"
)