#!/usr/bin/env python3
"""
Bronze ETL Package

A clean, modular framework for building bronze-layer ETL processes.
Based on the specifications in parquet_issues.md.
"""

from .aggregate_30min import (aggregate_30min_complete, aggregate_ohlcv_30min,
                              calculate_expected_bar_count,
                              convert_timestamp_to_datetime,
                              filter_to_30min_timestamps,
                              validate_bar_alignment)
from .config import (BATCH_CONFIG, DATE_END, DATE_END_PL, DATE_START,
                     DATE_START_PL, ETF_TICKERS, GPU_CONFIG, OUTPUT_PATHS,
                     RAW_PATHS, SCHEMA_PATHS, UNIVERSE, UNIVERSE_SET,
                     VALIDATION_CONFIG, get_output_path, get_raw_path,
                     get_schema_path, get_universe_for_data_type, is_etf,
                     validate_config)
from .etl_main import (load_raw_data, main, process_options_30min,
                       process_options_daily, process_stocks_30min,
                       process_stocks_daily, run_etl_pipeline, setup_logging,
                       validate_input_data)
from .loader import (benchmark_loading_performance, get_data_summary,
                     get_parquet_files, load_and_validate_sample,
                     load_parquet_batch_with_timing, load_parquet_with_timing,
                     save_parquet_with_timing, setup_gpu_acceleration)
from .parse_options import (add_option_fields, filter_options_by_universe,
                            parse_option_ticker, parse_options_complete,
                            validate_option_data_quality,
                            validate_option_parsing)
from .utils import (add_common_columns, apply_common_filters,
                    convert_timestamp_columns, filter_to_30min_timestamps,
                    filter_to_date_range, filter_to_market_days,
                    filter_to_market_hours, get_market_hours_info,
                    get_trading_days_count, is_30min_timestamp, is_market_day,
                    is_market_hours, normalize_numeric_columns,
                    round_timestamp_to_30min, validate_required_columns)
from .validate_schema import (load_schema, validate_data_quality,
                              validate_field_types, validate_required_columns,
                              validate_sample_data, validate_schema_complete,
                              validate_timestamp_sanity)

__version__ = "1.0.0"
__author__ = "Conviction AI Team"

__all__ = [
    # Config
    "UNIVERSE",
    "UNIVERSE_SET",
    "ETF_TICKERS",
    "DATE_START",
    "DATE_END",
    "DATE_START_PL",
    "DATE_END_PL",
    "RAW_PATHS",
    "OUTPUT_PATHS",
    "SCHEMA_PATHS",
    "GPU_CONFIG",
    "BATCH_CONFIG",
    "VALIDATION_CONFIG",
    "get_universe_for_data_type",
    "get_raw_path",
    "get_output_path",
    "get_schema_path",
    "is_etf",
    "validate_config",
    # Loader
    "setup_gpu_acceleration",
    "get_parquet_files",
    "load_parquet_with_timing",
    "load_parquet_batch_with_timing",
    "save_parquet_with_timing",
    "load_and_validate_sample",
    "benchmark_loading_performance",
    "get_data_summary",
    # Schema Validation
    "load_schema",
    "validate_required_columns",
    "validate_field_types",
    "validate_timestamp_sanity",
    "validate_data_quality",
    "validate_schema_complete",
    "validate_sample_data",
    # Options Parsing
    "parse_option_ticker",
    "validate_option_parsing",
    "add_option_fields",
    "filter_options_by_universe",
    "validate_option_data_quality",
    "parse_options_complete",
    # 30-Minute Aggregation
    "convert_timestamp_to_datetime",
    "filter_to_30min_timestamps",
    "aggregate_ohlcv_30min",
    "validate_bar_alignment",
    "calculate_expected_bar_count",
    "aggregate_30min_complete",
    # Utils
    "is_market_day",
    "is_market_hours",
    "filter_to_market_hours",
    "filter_to_market_days",
    "filter_to_date_range",
    "convert_timestamp_columns",
    "normalize_numeric_columns",
    "add_common_columns",
    "validate_required_columns",
    "get_trading_days_count",
    "get_market_hours_info",
    "round_timestamp_to_30min",
    "is_30min_timestamp",
    "filter_to_30min_timestamps",
    "apply_common_filters",
    # Main ETL
    "setup_logging",
    "validate_input_data",
    "load_raw_data",
    "process_stocks_daily",
    "process_stocks_30min",
    "process_options_daily",
    "process_options_30min",
    "run_etl_pipeline",
    "main",
]
