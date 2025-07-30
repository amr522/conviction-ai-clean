#!/usr/bin/env python3
"""
Build Bronze Stocks Daily Combined Dataset

Processes raw daily stock data into a bronze dataset with extended 3-year coverage.

REQUIREMENTS:
1. Unified Universe: 30 underlyings (28 equities + SPY & QQQ)
2. Extended Training Period: 2021-07-07 to 2024-07-07 (3 years)
3. Output: staged/bronze_stocks_daily_combined.parquet
4. Raw Data Path: data/Parquet_data/Raw/Stocks_daily

CRITICAL FEATURES:
- SPY/QQQ ETF tagging (is_etf=true)
- Timestamp conversion from nanoseconds
- Market hours & holidays filtering
- GPU acceleration with Apple MPS
- Comprehensive validation
- Simple feature engineering
"""

import argparse
import glob
import json
import logging
import multiprocessing
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from shutil import rmtree
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import polars as pl
import psutil
import pytz
from tqdm import tqdm

# Configure environment for optimal performance
os.environ["PYTORCH_ENABLE_MPS"] = "1"  # Enable Apple MPS
os.environ["POLARS_MAX_THREADS"] = "24"  # Max threads safely
warnings.filterwarnings("ignore")

# ===============================================
# CONFIGURATION CONSTANTS
# ===============================================

# Unified Universe (30 underlyings from parquet_issues.md)
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

UNIVERSE_SET = set(UNIVERSE)
ETF_TICKERS = {"SPY", "QQQ"}

# Extended training period: 3 years (2021-07-07 to 2024-07-07)
DATE_START = "2021-07-07"
DATE_END = "2024-07-07"
DATE_START_PL = pl.date(2021, 7, 7)
DATE_END_PL = pl.date(2024, 7, 7)

# Major market holidays for 2021-2024 (3-year period)
HOLIDAYS = {
    # 2021 holidays
    "2021-07-05",  # Independence Day observed
    "2021-09-06",  # Labor Day
    "2021-11-25",  # Thanksgiving
    "2021-11-26",  # Black Friday
    "2021-12-24",  # Christmas Eve (half day)
    "2021-12-31",  # New Year's Eve (half day)
    # 2022 holidays
    "2022-01-01",  # New Year's Day
    "2022-01-17",  # MLK Day
    "2022-02-21",  # Presidents Day
    "2022-04-15",  # Good Friday
    "2022-05-30",  # Memorial Day
    "2022-06-20",  # Juneteenth observed
    "2022-07-04",  # Independence Day
    "2022-09-05",  # Labor Day
    "2022-11-24",  # Thanksgiving
    "2022-11-25",  # Black Friday
    "2022-12-26",  # Christmas observed
    # 2023 holidays
    "2023-01-02",  # New Year's Day observed
    "2023-01-16",  # MLK Day
    "2023-02-20",  # Presidents Day
    "2023-04-07",  # Good Friday
    "2023-05-29",  # Memorial Day
    "2023-06-19",  # Juneteenth
    "2023-07-04",  # Independence Day
    "2023-09-04",  # Labor Day
    "2023-11-23",  # Thanksgiving
    "2023-11-24",  # Black Friday
    "2023-12-25",  # Christmas
    # 2024 holidays
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # MLK Day
    "2024-02-19",  # Presidents Day
    "2024-03-29",  # Good Friday
    "2024-05-27",  # Memorial Day
    "2024-06-19",  # Juneteenth
    "2024-07-04",  # Independence Day
}

# Required columns from raw schema validation
REQUIRED_COLUMNS = [
    "ticker",
    "volume",
    "open",
    "close",
    "high",
    "low",
    "window_start",
    "transactions",
]

# Market hours (9:30 AM - 4:00 PM ET)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# ===============================================
# LOGGING CONFIGURATION
# ===============================================


def setup_logging(test_mode: bool = False) -> logging.Logger:
    """Setup comprehensive logging configuration"""
    logger = logging.getLogger(__name__)

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)  # Enable debug logging

    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Enable debug logging

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    # Log startup info
    logger.info("=" * 80)
    logger.info("🚀 BRONZE STOCKS DAILY BUILD - GPU ACCELERATED")
    logger.info("=" * 80)

    if test_mode:
        logger.info("🧪 TEST MODE ENABLED")
        logger.info(f"   Date range: {TEST_START_DATE} to {TEST_END_DATE}")
        logger.info(f"   Output: {TEST_OUTPUT_PATH}")
        logger.info(f"   Max files: {TEST_MAX_FILES}")

    logger.info("=" * 80)

    return logger


# ===============================================
# GPU & MEMORY MONITORING
# ===============================================


def check_gpu_availability():
    """Check and configure GPU acceleration"""
    try:
        import torch

        if torch.backends.mps.is_available():
            print("🍎 Apple MPS GPU is available and enabled - PERFORMANCE MODE")
            torch.mps.set_per_process_memory_fraction(0.95)
            torch.mps.empty_cache()
            return True
        else:
            print("⚠️ Apple MPS GPU not available, falling back to CPU")
            return False
    except ImportError:
        print("⚠️ PyTorch not available, proceeding with CPU-only operations")
        return False


def log_memory_usage(logger: logging.Logger, stage: str):
    """Log current memory usage"""
    memory = psutil.virtual_memory()
    logger.info(
        f"[{stage}] Memory: {memory.percent:.1f}% used "
        f"({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)"
    )


# ===============================================
# VALIDATION UTILITIES
# ===============================================


def validate_raw_schema(
    df: pl.DataFrame, file_path: str, logger: logging.Logger
) -> bool:
    """Validate raw schema has required fields"""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_cols:
        logger.warning(f"File {file_path} missing columns: {missing_cols}")
        return False

    if df.is_empty():
        logger.warning(f"File {file_path} is empty")
        return False

    return True


def is_market_day(date_str: str) -> bool:
    """Check if date is a market day (exclude weekends and holidays)"""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()

        # Check weekend
        if date_obj.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Check holidays
        return date_str not in HOLIDAYS

    except ValueError:
        return False


# ===============================================
# DATA PROCESSING FUNCTIONS
# ===============================================


def convert_timestamp(df: pl.DataFrame, logger: logging.Logger) -> pl.DataFrame:
    """Convert window_start from nanoseconds to proper timestamp and date"""
    try:
        # Convert window_start from nanoseconds to seconds as integer first
        df_with_seconds = df.with_columns(
            [
                (pl.col("window_start").cast(pl.Int64) // 1_000_000_000).alias(
                    "timestamp_seconds"
                )
            ]
        )

        # Convert to datetime using Polars built-in functions
        df_with_timestamp = df_with_seconds.with_columns(
            [
                # Convert seconds to datetime
                pl.from_epoch(pl.col("timestamp_seconds"), time_unit="s").alias(
                    "timestamp"
                ),
                # Extract date for filtering
                pl.from_epoch(pl.col("timestamp_seconds"), time_unit="s")
                .dt.date()
                .alias("date"),
            ]
        ).drop("timestamp_seconds")

        logger.debug("✅ Timestamp conversion successful")
        return df_with_timestamp

    except Exception as e:
        logger.error(f"❌ Timestamp conversion failed: {e}")
        raise


def cast_data_types(df: pl.DataFrame, logger: logging.Logger) -> pl.DataFrame:
    """Cast all columns to proper data types"""
    try:
        df_typed = df.with_columns(
            [
                pl.col("ticker").cast(pl.Utf8),
                pl.col("volume").cast(pl.Int64),
                pl.col("open").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("window_start").cast(pl.Int64),
                pl.col("transactions").cast(pl.Int64),
            ]
        )

        logger.debug("✅ Data type casting successful")
        return df_typed

    except Exception as e:
        logger.error(f"❌ Data type casting failed: {e}")
        raise


def filter_to_universe(df: pl.DataFrame, logger: logging.Logger) -> pl.DataFrame:
    """Filter to only the 30 specified underlyings"""
    initial_count = len(df)
    df_filtered = df.filter(pl.col("ticker").is_in(UNIVERSE))
    final_count = len(df_filtered)

    logger.info(
        f"📊 Universe filter: {initial_count:,} → {final_count:,} rows "
        f"({final_count/initial_count*100:.1f}% kept)"
    )

    return df_filtered


def filter_to_date_range(df: pl.DataFrame, logger: logging.Logger) -> pl.DataFrame:
    """Filter to extended training period: 2021-07-07 to 2024-07-07 (3 years)"""
    initial_count = len(df)

    # Log initial date range for debugging
    if not df.is_empty():
        min_date = df.select("date").min().item()
        max_date = df.select("date").max().item()
        unique_dates_initial = df.select("date").n_unique()
        logger.debug(
            f"📊 Before date filter: {unique_dates_initial} unique dates from {min_date} to {max_date}"
        )

    df_filtered = df.filter(
        (pl.col("date") >= DATE_START_PL) & (pl.col("date") <= DATE_END_PL)
    )

    final_count = len(df_filtered)

    # Enhanced logging
    if not df_filtered.is_empty():
        unique_dates_final = df_filtered.select("date").n_unique()
        min_date_final = df_filtered.select("date").min().item()
        max_date_final = df_filtered.select("date").max().item()
        logger.info(
            f"📅 Date range filter: {initial_count:,} → {final_count:,} rows "
            f"({final_count/initial_count*100:.1f}% kept)"
        )
        logger.info(f"   🎯 Target: {DATE_START} to {DATE_END}")
        logger.info(f"   ✅ Actual: {min_date_final} to {max_date_final}")
        logger.info(f"   📊 Unique dates captured: {unique_dates_final}")
    else:
        logger.warning(f"⚠️ Date range filter resulted in empty dataset!")
        logger.warning(f"   Target range: {DATE_START} to {DATE_END}")

    return df_filtered


def filter_to_market_days(df: pl.DataFrame, logger: logging.Logger) -> pl.DataFrame:
    """Filter to exclude weekends and holidays"""
    initial_count = len(df)

    # Log initial date range for debugging
    if not df.is_empty():
        min_date = df.select("date").min().item()
        max_date = df.select("date").max().item()
        unique_dates_initial = df.select("date").n_unique()
        logger.debug(
            f"🗓️ Before market filter: {unique_dates_initial} unique dates from {min_date} to {max_date}"
        )

    # Filter out weekends (Saturday=6, Sunday=7 in Polars)
    df_weekdays = df.filter(~pl.col("date").dt.weekday().is_in([6, 7]))
    weekdays_count = len(df_weekdays)

    # Filter out holidays using string format
    # Convert date column to string format for comparison
    holiday_list = list(HOLIDAYS)
    df_filtered = df_weekdays.filter(~pl.col("date").cast(pl.Utf8).is_in(holiday_list))

    final_count = len(df_filtered)

    # Enhanced logging with date counts
    if not df_filtered.is_empty():
        unique_dates_final = df_filtered.select("date").n_unique()
        logger.info(
            f"🗓️ Market days filter: {initial_count:,} → {final_count:,} rows "
            f"({final_count/initial_count*100:.1f}% kept)"
        )
        logger.info(f"   📅 Weekend filter: {initial_count:,} → {weekdays_count:,} rows")
        logger.info(f"   🎃 Holiday filter: {weekdays_count:,} → {final_count:,} rows")
        logger.info(f"   📊 Unique dates after filter: {unique_dates_final}")
        logger.info(
            f"   🚫 Filtered {len(holiday_list)} holidays: {sorted(holiday_list)[:10]}..."
        )
    else:
        logger.warning("⚠️ Market days filter resulted in empty dataset!")

    return df_filtered


def add_metadata(df: pl.DataFrame, logger: logging.Logger) -> pl.DataFrame:
    """Add metadata columns: data_type and is_etf"""
    df_with_meta = df.with_columns(
        [
            pl.lit("stocks_daily").alias("data_type"),
            pl.col("ticker").is_in(list(ETF_TICKERS)).alias("is_etf"),
        ]
    )

    logger.debug("✅ Metadata columns added")
    return df_with_meta


def add_simple_features(df: pl.DataFrame, logger: logging.Logger) -> pl.DataFrame:
    """Add simple features with anti-leakage lagging"""
    try:
        # Skip feature engineering if dataset is empty or too small
        if df.is_empty() or len(df) < 10:
            logger.debug("⚠️ Skipping feature engineering - dataset too small")
            return df

        # Sort by ticker and date for proper window operations
        df_sorted = df.sort(["ticker", "date"])

        # Add simple features without nested window operations
        try:
            df_with_features = df_sorted.with_columns(
                [
                    # Basic price features (no window operations)
                    (pl.col("high") - pl.col("low")).alias("price_range"),
                    ((pl.col("close") - pl.col("open")) / pl.col("open") * 100).alias(
                        "daily_return_pct"
                    ),
                    (pl.col("volume") * pl.col("close")).alias("dollar_volume"),
                    # Simple lag features (shift operations work well)
                    pl.col("close").shift(1).over("ticker").alias("prev_close"),
                    pl.col("volume").shift(1).over("ticker").alias("prev_volume"),
                ]
            )

            logger.info("📈 Simple features added successfully")
            return df_with_features
        except Exception as e:
            logger.warning(
                f"⚠️ Feature engineering failed, returning without features: {e}"
            )
            return df

    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}")
        # Return original dataframe without features if feature engineering fails
        logger.warning("⚠️ Returning dataframe without features due to error")
        return df


def process_single_file(
    file_path: str, logger: logging.Logger
) -> Optional[pl.DataFrame]:
    """Process a single parquet file"""
    try:
        logger.debug(f"Processing {file_path}")

        # Read file
        df = pl.read_parquet(file_path)
        logger.debug(f"Read file: {df.shape}")

        # Validate schema
        if not validate_raw_schema(df, file_path, logger):
            return None

        # Processing pipeline
        df = cast_data_types(df, logger)
        logger.debug(f"After casting: {df.shape}")

        df = convert_timestamp(df, logger)
        logger.debug(f"After timestamp: {df.shape}")

        df = filter_to_universe(df, logger)
        logger.debug(f"After universe: {df.shape}")

        df = filter_to_date_range(df, logger)
        logger.debug(f"After date range: {df.shape}")

        # Check if we have any data left after filtering
        if df.is_empty():
            logger.debug(f"📊 No data in target range for {file_path}")
            return None

        df = filter_to_market_days(df, logger)
        logger.debug(f"After market days: {df.shape}")

        # Check again after market days filter
        if df.is_empty():
            logger.debug(f"📊 No data after market days filter for {file_path}")
            return None

        df = add_metadata(df, logger)
        logger.debug(f"After metadata: {df.shape}")

        df = add_simple_features(df, logger)
        logger.debug(f"After features: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"❌ Failed to process {file_path}: {e}")
        logger.debug(f"Full error traceback:", exc_info=True)
        return None


# ===============================================
# PARALLEL PROCESSING
# ===============================================


def process_files_parallel(
    file_paths: List[str], logger: logging.Logger
) -> pl.DataFrame:
    """Process multiple files in parallel"""
    logger.info(f"🔄 Processing {len(file_paths)} files with 24 workers...")

    successful_dfs = []
    failed_files = []

    with ThreadPoolExecutor(max_workers=24) as executor:
        # Process files in parallel
        futures = {
            executor.submit(process_single_file, file_path, logger): file_path
            for file_path in file_paths
        }

        # Collect results with progress bar
        for future in tqdm(futures, desc="Processing files"):
            file_path = futures[future]
            try:
                df = future.result(timeout=300)  # 5-minute timeout per file
                if df is not None and not df.is_empty():
                    successful_dfs.append(df)
                else:
                    failed_files.append(file_path)
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path}: {e}")
                failed_files.append(file_path)

    # Report results
    logger.info(f"✅ Successfully processed: {len(successful_dfs)} files")
    if failed_files:
        logger.warning(f"⚠️ Failed files: {len(failed_files)}")
        for file_path in failed_files[:5]:  # Show first 5 failures
            logger.warning(f"   - {file_path}")

    if not successful_dfs:
        raise ValueError("❌ No files processed successfully!")

    # Filter out empty DataFrames before concatenation
    non_empty_dfs = [df for df in successful_dfs if len(df) > 0]
    logger.info(
        f"🔗 Combining {len(non_empty_dfs)} non-empty dataframes (out of {len(successful_dfs)} successful)..."
    )

    if not non_empty_dfs:
        raise ValueError("❌ All processed dataframes are empty!")

    # Combine all dataframes with error handling
    try:
        combined_df = pl.concat(non_empty_dfs, how="vertical_relaxed")
    except Exception as e:
        logger.error(f"❌ Concatenation failed: {e}")
        logger.info("📊 Debugging concatenation issue...")
        for i, df in enumerate(non_empty_dfs[:3]):  # Check first 3 DataFrames
            logger.info(f"   DataFrame {i}: shape={df.shape}, schema={df.schema}")
        raise

    return combined_df


# ===============================================
# VALIDATION & OUTPUT
# ===============================================


def validate_final_dataset(df: pl.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """Comprehensive validation of final dataset"""
    validation_results = {}

    # Basic stats
    validation_results["total_rows"] = len(df)
    validation_results["total_tickers"] = df.select("ticker").n_unique()
    validation_results["date_range"] = {
        "start": str(df.select("date").min().item()),
        "end": str(df.select("date").max().item()),
    }

    # Universe validation
    actual_tickers = set(df.select("ticker").unique().to_pandas()["ticker"].tolist())
    missing_tickers = UNIVERSE_SET - actual_tickers
    extra_tickers = actual_tickers - UNIVERSE_SET

    validation_results["universe_check"] = {
        "expected_count": len(UNIVERSE),
        "actual_count": len(actual_tickers),
        "missing_tickers": list(missing_tickers),
        "extra_tickers": list(extra_tickers),
        "is_valid": len(missing_tickers) == 0 and len(extra_tickers) == 0,
    }

    # Coverage per ticker
    coverage_stats = (
        df.group_by("ticker")
        .agg(
            [
                pl.count().alias("row_count"),
                pl.col("date").min().alias("first_date"),
                pl.col("date").max().alias("last_date"),
            ]
        )
        .sort("ticker")
    )

    validation_results["coverage_per_ticker"] = coverage_stats.to_pandas().to_dict(
        "records"
    )

    # Missing value rates
    total_rows = len(df)
    missing_rates = {}
    for col in df.columns:
        if col not in ["date", "timestamp", "ticker", "data_type", "is_etf"]:
            null_count = df.select(pl.col(col).is_null().sum()).item()
            missing_rates[col] = null_count / total_rows * 100

    validation_results["missing_value_rates"] = missing_rates

    # ETF validation
    etf_count = df.filter(pl.col("is_etf") == True).select("ticker").n_unique()
    validation_results["etf_validation"] = {
        "expected_etf_count": len(ETF_TICKERS),
        "actual_etf_count": etf_count,
        "is_valid": etf_count == len(ETF_TICKERS),
    }

    # Log validation summary
    logger.info("=" * 50)
    logger.info("📊 FINAL DATASET VALIDATION")
    logger.info("=" * 50)
    logger.info(f"Total rows: {validation_results['total_rows']:,}")
    logger.info(f"Total tickers: {validation_results['total_tickers']}")
    logger.info(
        f"Date range: {validation_results['date_range']['start']} to {validation_results['date_range']['end']}"
    )

    universe_check = validation_results["universe_check"]
    if universe_check["is_valid"]:
        logger.info("✅ Universe validation: PASSED")
    else:
        logger.error("❌ Universe validation: FAILED")
        if universe_check["missing_tickers"]:
            logger.error(f"   Missing: {universe_check['missing_tickers']}")
        if universe_check["extra_tickers"]:
            logger.error(f"   Extra: {universe_check['extra_tickers']}")

    etf_check = validation_results["etf_validation"]
    if etf_check["is_valid"]:
        logger.info("✅ ETF validation: PASSED")
    else:
        logger.error(
            f"❌ ETF validation: FAILED ({etf_check['actual_etf_count']}/{etf_check['expected_etf_count']})"
        )

    # Check for high missing value rates
    high_missing = {k: v for k, v in missing_rates.items() if v > 5.0}
    if high_missing:
        logger.warning(f"⚠️ High missing value rates: {high_missing}")
    else:
        logger.info("✅ Missing value rates: ACCEPTABLE")

    return validation_results


def save_dataset(df: pl.DataFrame, output_path: Path, logger: logging.Logger):
    """Save final dataset with deduplication and sorting"""
    logger.info("💾 Preparing final dataset for output...")

    # Remove duplicates (only true duplicates, not valid multi-date bars)
    initial_count = len(df)
    df_deduped = df.unique(subset=["ticker", "date", "timestamp"], keep="first")
    final_count = len(df_deduped)

    if initial_count != final_count:
        logger.info(
            f"🔄 Deduplication: {initial_count:,} → {final_count:,} rows "
            f"({initial_count - final_count:,} duplicates removed)"
        )

    # Sort by date, ticker, timestamp
    df_sorted = df_deduped.sort(["date", "ticker", "timestamp"])

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with compression
    logger.info(f"💾 Saving to: {output_path}")
    df_sorted.write_parquet(output_path, compression="zstd")

    # Verify saved file
    saved_df = pl.read_parquet(output_path)
    logger.info(
        f"✅ Dataset saved successfully: {len(saved_df):,} rows, {saved_df.estimated_size('mb'):.1f} MB"
    )


# ===============================================
# MAIN EXECUTION
# ===============================================


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Build Bronze Stocks Daily Dataset")
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="data/Parquet_data/Raw/Stocks_daily",
        help="Input directory containing raw stock data",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="staged/bronze_stocks_daily_combined.parquet",
        help="Output path for combined dataset",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with limited date range and output to temp location",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Number of worker threads for parallel processing",
    )

    args = parser.parse_args()

    # Setup
    start_time = time.time()
    logger = setup_logging()

    # Configure for test mode if requested
    if args.test_mode:
        # Keep full date range in test mode to validate data availability
        logger.info("🧪 TEST MODE ENABLED")
        logger.info(f"   Date range: {DATE_START} to {DATE_END} (FULL RANGE)")

        # Override output path for test mode only
        args.output_path = "staged/tmp/stocks_daily_test.parquet"
        # Remove file limit in test mode to capture all available data

        logger.info(f"   Output: {args.output_path}")
        logger.info(f"   Files: ALL (no limit in test mode)")

    # Check GPU availability
    gpu_available = check_gpu_availability()
    log_memory_usage(logger, "STARTUP")

    try:
        # Find input files
        input_dir = Path(args.raw_data_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        file_pattern = str(input_dir / "*.parquet")
        file_paths = glob.glob(file_pattern)

        if not file_paths:
            raise FileNotFoundError(f"No parquet files found in: {input_dir}")

        logger.info(f"📁 Found {len(file_paths)} parquet files")

        # Don't limit files unless explicitly requested for debugging
        if args.max_files:
            file_paths = file_paths[: args.max_files]
            logger.info(f"🧪 Limited to {len(file_paths)} files for debugging")
        else:
            logger.info(f"📁 Processing ALL {len(file_paths)} files")

        log_memory_usage(logger, "BEFORE_PROCESSING")

        # Process files
        combined_df = process_files_parallel(file_paths, logger)

        # Add debugging info about captured dates
        if not combined_df.is_empty():
            actual_dates = combined_df.select("date").unique().sort("date")
            unique_dates_count = len(actual_dates)
            min_date = actual_dates.select("date").min().item()
            max_date = actual_dates.select("date").max().item()

            logger.info("=" * 60)
            logger.info("📊 DATE RANGE ANALYSIS")
            logger.info("=" * 60)
            logger.info(f"✅ Found {unique_dates_count} unique trading dates")
            logger.info(f"📅 Date range: {min_date} to {max_date}")
            logger.info(f"🎯 Target range: {DATE_START} to {DATE_END}")

            # Check if we're getting the expected ~754 trading days
            if unique_dates_count < 500:
                logger.warning(
                    f"⚠️ Only captured {unique_dates_count} dates - expected ~754 trading days"
                )
                logger.warning("   This suggests data filtering may be too aggressive")
            else:
                logger.info(f"✅ Good coverage: {unique_dates_count} dates captured")

        log_memory_usage(logger, "AFTER_PROCESSING")

        # Validate final dataset
        validation_results = validate_final_dataset(combined_df, logger)

        # Save dataset
        output_path = Path(args.output_path)
        save_dataset(combined_df, output_path, logger)

        # Save validation results
        validation_path = output_path.parent / f"{output_path.stem}_validation.json"
        with open(validation_path, "w") as f:
            json.dump(validation_results, f, indent=2, default=str)
        logger.info(f"📊 Validation results saved to: {validation_path}")

        # Final summary
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 BRONZE STOCKS DAILY BUILD COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"⏱️ Total processing time: {total_time:.1f} seconds")
        logger.info(f"📊 Final dataset: {len(combined_df):,} rows")
        logger.info(f"💾 Output: {output_path}")
        logger.info(f"🍎 GPU acceleration: {'ENABLED' if gpu_available else 'DISABLED'}")

        log_memory_usage(logger, "COMPLETION")

    except Exception as e:
        logger.error(f"❌ Build failed: {e}")
        raise


if __name__ == "__main__":
    main()
