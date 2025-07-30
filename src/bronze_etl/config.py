#!/usr/bin/env python3
"""
Bronze ETL Configuration

Defines unified universe, date ranges, and raw data paths for all bronze ETL processes.
Based on the specifications in parquet_issues.md.
"""

from pathlib import Path
from typing import Any, Dict, List

import polars as pl

# ---------------------------------------------------------------
# 1. UNIFIED UNDERLYING UNIVERSE (30 underlyings)
# ---------------------------------------------------------------

# 30 underlyings (28 equities + SPY & QQQ) as specified in parquet_issues.md
# Note: SPY and QQQ are treated as regular tickers in raw data, tagged as ETFs downstream
UNIVERSE = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",  # Tech
    "JPM",
    "GS",
    "MS",
    "BAC",
    "V",
    "MA",  # Financial
    "TSLA",
    "NFLX",
    "DIS",
    "WMT",  # Consumer
    "JNJ",
    "PFE",
    "MRK",
    "UNH",
    "ABBV",  # Healthcare
    "XOM",
    "CVX",  # Energy
    "HOOD",
    "PLTR",
    "MSTR",
    "COIN",
    "AMD",  # High IV
    "SPY",
    "QQQ",  # ETFs (treated as regular tickers)
]

# Convert to set for faster lookups
UNIVERSE_SET = set(UNIVERSE)

# ETF tickers for downstream workflows
ETF_TICKERS = {"SPY", "QQQ"}

# ---------------------------------------------------------------
# 2. TRAINING PERIOD (2021-07-07 to 2024-07-07 - 3-year training period)
# ---------------------------------------------------------------

DATE_START = "2021-07-07"
DATE_END = "2024-07-07"

# Polars date objects for efficient filtering
DATE_START_PL = pl.date(2021, 7, 7)
DATE_END_PL = pl.date(2024, 7, 7)

# ---------------------------------------------------------------
# 3. RAW DATA PATHS
# ---------------------------------------------------------------

RAW_PATHS = {
    "news": "data/Parquet_data/Raw/news",
    "options_minute": "data/Parquet_data/Raw/option_minute",
    "options_daily": "data/Parquet_data/Raw/options_daily",
    "stocks_daily": "data/Parquet_data/Raw/Stocks_daily",
    "stocks_minute": "data/Parquet_data/Raw/stocks_minute",
    "dxy": "data/Parquet_data/Raw/DXY.csv",
    "fred": "data/Parquet_data/Raw/FRED.csv",
    "vix": "data/Parquet_data/vix_data.parquet",
}

# ---------------------------------------------------------------
# 4. OUTPUT PATHS
# ---------------------------------------------------------------

OUTPUT_PATHS = {
    "stocks_daily": "staged/bronze_stocks_daily.parquet",
    "stocks_30min": "staged/bronze_stocks_30min.parquet",
    "options_daily": "staged/bronze_options_daily.parquet",
    "options_30min": "staged/bronze_options_30min.parquet",
}

# ---------------------------------------------------------------
# 5. SCHEMA PATHS
# ---------------------------------------------------------------

SCHEMA_PATHS = {
    "stocks_daily": "schemas/stocks_daily_raw.json",
    "stocks_minute": "schemas/stocks_minute_raw.json",
    "options_daily": "schemas/options_daily_raw.json",
    "options_30min": "schemas/options_30min_raw.json",
}

# ---------------------------------------------------------------
# 6. MARKET HOURS & CALENDAR
# ---------------------------------------------------------------

# US Market hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Complete US market holidays 2021-2024
HOLIDAYS = [
    # 2021 Holidays
    "2021-01-01",
    "2021-01-18",
    "2021-02-15",
    "2021-04-02",
    "2021-05-31",
    "2021-07-05",
    "2021-09-06",
    "2021-11-25",
    "2021-11-26",
    "2021-12-24",
    # 2022 Holidays
    "2022-01-17",
    "2022-02-21",
    "2022-04-15",
    "2022-05-30",
    "2022-06-20",
    "2022-07-04",
    "2022-09-05",
    "2022-11-24",
    "2022-12-26",
    # 2023 Holidays
    "2023-01-02",
    "2023-01-16",
    "2023-02-20",
    "2023-04-07",
    "2023-05-29",
    "2023-06-19",
    "2023-07-04",
    "2023-09-04",
    "2023-11-23",
    "2023-12-25",
    # 2024 Holidays
    "2024-01-01",
    "2024-01-15",
    "2024-02-19",
    "2024-03-29",
    "2024-05-27",
    "2024-06-19",
    "2024-07-04",
    "2024-09-02",
    "2024-11-28",
    "2024-12-25",
]

HOLIDAYS_SET = set(HOLIDAYS)

# ---------------------------------------------------------------
# 7. PERFORMANCE CONFIGURATION
# ---------------------------------------------------------------

# GPU/CPU configuration
CPU_THREADS = max(1, 24 // 4)  # Throttled to prevent system freezes
GPU_NUM_WORKERS = 24  # GPU workers for M2 Ultra
GPU_CONFIG = {
    "use_gpu": True,
    "device": "0",
    "memory_fraction": 0.90,
    "memory_growth": True,
    "strategy": "memory_efficient",
}

# Batch processing configuration
BATCH_CONFIG = {"batch_size": 20, "max_workers": 12, "chunk_size": 500000}

# ---------------------------------------------------------------
# 8. VALIDATION CONFIGURATION
# ---------------------------------------------------------------

VALIDATION_CONFIG = {
    "expected_bars_per_day": 13,  # 30-minute bars per trading day
    "min_bars_required": 10,  # Minimum bars for valid day
    "max_missing_rate": 0.01,  # 1% tolerance for missing data
    "timestamp_tolerance_seconds": 60,  # 1 minute tolerance for timestamp alignment
}

# ---------------------------------------------------------------
# 9. HELPER FUNCTIONS
# ---------------------------------------------------------------


def get_universe_for_data_type(data_type: str) -> List[str]:
    """Get the appropriate universe for a given data type"""
    if data_type in ["stocks_daily", "stocks_30min"]:
        return UNIVERSE
    elif data_type in ["options_daily", "options_30min"]:
        # For options, we include all options where underlying is in our universe
        return UNIVERSE  # The filtering happens in the parsing logic
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def get_raw_path(data_type: str) -> str:
    """Get the raw data path for a given data type"""
    path_mapping = {
        "stocks_daily": RAW_PATHS["stocks_daily"],
        "stocks_30min": RAW_PATHS["stocks_minute"],  # 1-min data for 30-min aggregation
        "options_daily": RAW_PATHS["options_daily"],
        "options_30min": RAW_PATHS[
            "options_minute"
        ],  # 1-min data for 30-min aggregation
    }

    if data_type not in path_mapping:
        raise ValueError(f"Unknown data type: {data_type}")

    return path_mapping[data_type]


def get_schema_path(data_type: str) -> str:
    """Get the schema path for a given data type"""
    if data_type not in SCHEMA_PATHS:
        raise ValueError(f"Unknown data type: {data_type}")

    return SCHEMA_PATHS[data_type]


def get_output_path(data_type: str) -> str:
    """Get the output path for a given data type"""
    if data_type not in OUTPUT_PATHS:
        raise ValueError(f"Unknown data type: {data_type}")

    return OUTPUT_PATHS[data_type]


def is_etf(ticker: str) -> bool:
    """Check if a ticker is an ETF"""
    return ticker in ETF_TICKERS


def validate_config() -> Dict[str, Any]:
    """Validate configuration and return status"""
    validation_results = {
        "universe_size": len(UNIVERSE),
        "date_range_days": 366,  # Approximate days between 2023-07-07 and 2024-07-07
        "raw_paths_exist": {},
        "schema_paths_exist": {},
        "output_dirs_creatable": {},
    }

    # Check raw paths
    for data_type, path in RAW_PATHS.items():
        validation_results["raw_paths_exist"][data_type] = Path(path).exists()

    # Check schema paths
    for data_type, path in SCHEMA_PATHS.items():
        validation_results["schema_paths_exist"][data_type] = Path(path).exists()

    # Check output directories
    for data_type, path in OUTPUT_PATHS.items():
        output_dir = Path(path).parent
        validation_results["output_dirs_creatable"][data_type] = (
            output_dir.exists() or output_dir.parent.exists()
        )

    return validation_results
