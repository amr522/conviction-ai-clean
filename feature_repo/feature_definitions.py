#!/usr/bin/env python3
"""
Feast feature view definitions
"""
from datetime import timedelta

from entities import ticker
from feast import Feature, FeatureView, FileSource, ValueType

# Stocks 30-minute features
stocks_30min_source = FileSource(
    path="staged/stocks_30min_clean.parquet",
    event_timestamp_column="timestamp",
    created_timestamp_column="timestamp",
)

stocks_30min_fv = FeatureView(
    name="stocks_30min",
    entities=["ticker"],
    ttl=timedelta(hours=1),
    features=[
        Feature(name="open", dtype=ValueType.FLOAT),
        Feature(name="high", dtype=ValueType.FLOAT),
        Feature(name="low", dtype=ValueType.FLOAT),
        Feature(name="close", dtype=ValueType.FLOAT),
        Feature(name="volume", dtype=ValueType.INT64),
        Feature(name="vwap", dtype=ValueType.FLOAT),
        Feature(name="returns", dtype=ValueType.FLOAT),
        Feature(name="volatility", dtype=ValueType.FLOAT),
    ],
    batch_source=stocks_30min_source,
    tags={"team": "conviction-ai", "type": "stocks"},
)

# Options 30-minute features
options_30min_source = FileSource(
    path="staged/options_30min_clean.parquet",
    event_timestamp_column="timestamp",
    created_timestamp_column="timestamp",
)

options_30min_fv = FeatureView(
    name="options_30min",
    entities=["ticker"],
    ttl=timedelta(hours=1),
    features=[
        Feature(name="opt30_open", dtype=ValueType.FLOAT),
        Feature(name="opt30_high", dtype=ValueType.FLOAT),
        Feature(name="opt30_low", dtype=ValueType.FLOAT),
        Feature(name="opt30_close", dtype=ValueType.FLOAT),
        Feature(name="opt30_volume", dtype=ValueType.INT64),
        Feature(name="opt30_call_flow", dtype=ValueType.INT64),
        Feature(name="opt30_put_flow", dtype=ValueType.INT64),
        Feature(name="opt30_flow_divergence", dtype=ValueType.FLOAT),
        Feature(name="opt30_gamma", dtype=ValueType.FLOAT),
        Feature(name="opt30_net_gamma", dtype=ValueType.FLOAT),
        Feature(name="opt30_gamma_squeeze", dtype=ValueType.BOOL),
        Feature(name="opt30_implied_volatility", dtype=ValueType.FLOAT),
        Feature(name="opt30_delta", dtype=ValueType.FLOAT),
        Feature(name="opt30_moneyness", dtype=ValueType.FLOAT),
    ],
    batch_source=options_30min_source,
    tags={"team": "conviction-ai", "type": "options"},
)

# Stocks daily features
stocks_daily_source = FileSource(
    path="staged/stocks_daily_clean.parquet",
    event_timestamp_column="timestamp",
    created_timestamp_column="timestamp",
)

stocks_daily_fv = FeatureView(
    name="stocks_daily",
    entities=["ticker"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="open", dtype=ValueType.FLOAT),
        Feature(name="high", dtype=ValueType.FLOAT),
        Feature(name="low", dtype=ValueType.FLOAT),
        Feature(name="close", dtype=ValueType.FLOAT),
        Feature(name="volume", dtype=ValueType.INT64),
        Feature(name="adj_close", dtype=ValueType.FLOAT),
        Feature(name="returns", dtype=ValueType.FLOAT),
        Feature(name="volatility_30d", dtype=ValueType.FLOAT),
        Feature(name="sma_20", dtype=ValueType.FLOAT),
        Feature(name="rsi_14", dtype=ValueType.FLOAT),
    ],
    batch_source=stocks_daily_source,
    tags={"team": "conviction-ai", "type": "stocks"},
)

# Options daily features
options_daily_source = FileSource(
    path="staged/options_daily_clean.parquet",
    event_timestamp_column="timestamp",
    created_timestamp_column="timestamp",
)

options_daily_fv = FeatureView(
    name="options_daily",
    entities=["ticker"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="optd_close", dtype=ValueType.FLOAT),
        Feature(name="optd_volume", dtype=ValueType.INT64),
        Feature(name="optd_moneyness", dtype=ValueType.FLOAT),
        Feature(name="optd_iv30", dtype=ValueType.FLOAT),
        Feature(name="optd_hv30", dtype=ValueType.FLOAT),
        Feature(name="optd_vrp_30d", dtype=ValueType.FLOAT),
        Feature(name="optd_iv_percentile", dtype=ValueType.FLOAT),
        Feature(name="optd_vol_spike", dtype=ValueType.BOOL),
        Feature(name="optd_put_call_ratio", dtype=ValueType.FLOAT),
    ],
    batch_source=options_daily_source,
    tags={"team": "conviction-ai", "type": "options"},
)
