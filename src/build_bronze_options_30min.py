#!/usr/bin/env python3
"""
Bronze 30-Minute Options ETL Pipeline

Goal: Produce `staged/bronze_options_30min.parquet` covering 2021-07-07 → 2024-07-07
for all option chains in our 30-symbol universe. Parse raw codes, aggregate 1-min →
true 30-min OHLCV bars (13 bars/day), apply simple features at underlying level,
and enforce all quality rules.
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import psutil
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
UNIVERSE = [
    "AAPL",
    "ABBV",
    "AMD",
    "AMZN",
    "COIN",
    "GOOG",
    "META",
    "MSFT",
    "NFLX",
    "NVDA",
    "PLTR",
    "PYPL",
    "RIVN",
    "RKLB",
    "ROKU",
    "SQ",
    "TSLA",
    "TSM",
    "UBER",
    "ZM",
    "QQQ",
    "SPY",
    "GLD",
    "IWM",
    "XLF",
    "XLK",
    "TLT",
    "EEM",
    "FXI",
    "EWZ",
]

DATE_START = "2021-07-07"
DATE_END = "2024-07-07"

# Market holidays (US)
US_HOLIDAYS = [
    "2021-07-05",
    "2021-09-06",
    "2021-11-25",
    "2021-11-26",
    "2021-12-24",
    "2021-12-31",
    "2022-01-01",
    "2022-01-17",
    "2022-02-21",
    "2022-04-15",
    "2022-05-30",
    "2022-06-20",
    "2022-07-04",
    "2022-09-05",
    "2022-11-24",
    "2022-11-25",
    "2022-12-26",
    "2023-01-02",
    "2023-01-16",
    "2023-02-20",
    "2023-04-07",
    "2023-05-29",
    "2023-06-19",
    "2023-07-04",
    "2023-09-04",
    "2023-11-23",
    "2023-11-24",
    "2023-12-25",
    "2024-01-01",
    "2024-01-15",
    "2024-02-19",
    "2024-03-29",
    "2024-05-27",
    "2024-06-19",
    "2024-07-04",
]


def get_memory_usage():
    """Get current memory usage."""
    memory = psutil.virtual_memory()
    return memory.percent, memory.used / (1024**3), memory.total / (1024**3)


def parse_option_codes_vectorized(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parse option codes vectorially using Polars expressions.
    Much faster than row-by-row parsing for 571M records.
    """
    logger.info("🚀 Vectorized option code parsing...")

    return (
        df.with_columns(
            [
                # Keep raw code for reference
                pl.col("ticker").alias("raw_code"),
                # Remove O: prefix
                pl.col("ticker").str.slice(2).alias("clean_code"),
            ]
        )
        .with_columns(
            [
                # Extract underlying (everything before the date pattern)
                pl.col("clean_code").str.extract(r"^([A-Z]+)", 1).alias("underlying"),
                # Extract date part (6 digits: YYMMDD)
                pl.col("clean_code")
                .str.extract(r"([0-9]{6})[CP]", 1)
                .alias("date_part"),
                # Extract option type (C or P)
                pl.col("clean_code")
                .str.extract(r"[0-9]{6}([CP])", 1)
                .alias("option_type"),
                # Extract strike (digits after C/P)
                pl.col("clean_code")
                .str.extract(r"[0-9]{6}[CP]([0-9]+)", 1)
                .alias("strike_str"),
            ]
        )
        .with_columns(
            [
                # Convert date to proper format (20YYMMDD)
                pl.concat_str([pl.lit("20"), pl.col("date_part")]).alias("exp_date"),
                # Convert strike to numeric (divide by 1000)
                (pl.col("strike_str").cast(pl.Int64) / 1000.0).alias("strike"),
            ]
        )
        .filter(
            # Filter out any failed parses (nulls)
            pl.col("underlying").is_not_null()
            & pl.col("exp_date").is_not_null()
            & pl.col("option_type").is_not_null()
            & pl.col("strike").is_not_null()
        )
        .drop(["clean_code", "date_part", "strike_str", "ticker"])
    )
    try:
        if not raw_code.startswith("O:"):
            return None

        # Remove O: prefix
        parts = raw_code[2:]

        # Find the date part (6 digits YYMMDD)
        # Look for the pattern where 6 digits are followed by C or P
        for i in range(
            len(parts) - 14
        ):  # Need at least 14 chars after start (6 date + 1 type + 8 strike)
            substr = parts[i : i + 6]
            if substr.isdigit() and i + 6 < len(parts) and parts[i + 6] in ["C", "P"]:
                underlying = parts[:i]
                date_str = substr
                option_type = parts[i + 6]
                strike_str = parts[i + 7 :]

                if len(strike_str) >= 8 and strike_str.isdigit():
                    # Convert to proper formats
                    exp_date = f"20{date_str}"
                    strike = int(strike_str) / 1000.0

                    return {
                        "underlying": underlying,
                        "exp_date": exp_date,
                        "option_type": option_type,
                        "strike": strike,
                    }

        return None

    except Exception:
        return None


def filter_market_hours(df: pl.DataFrame) -> pl.DataFrame:
    """Filter to market hours (9:30 AM - 4:00 PM ET)."""
    return df.filter(
        (pl.col("timestamp").dt.hour() >= 9) & (pl.col("timestamp").dt.hour() < 16)
        | (
            (pl.col("timestamp").dt.hour() == 9)
            & (pl.col("timestamp").dt.minute() >= 30)
        )
    )


def filter_holidays(df: pl.DataFrame) -> pl.DataFrame:
    """Filter out US holidays and weekends."""
    # Convert holiday strings to datetime dates
    from datetime import date

    holiday_dates = [date.fromisoformat(h) for h in US_HOLIDAYS]

    return df.filter(
        # Remove weekends (Monday=1, Sunday=7)
        (pl.col("timestamp").dt.weekday() <= 5)
        &
        # Remove holidays
        (~pl.col("timestamp").dt.date().is_in(holiday_dates))
    )


def read_and_parse_minute_data(
    raw_data_dir: str, test_mode: bool = False
) -> pl.DataFrame:
    """Read all minute files and parse raw option codes."""
    logger.info("📂 Reading minute-level option data...")

    # Find all parquet files
    raw_dir = Path(raw_data_dir)
    parquet_files = list(raw_dir.glob("*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {raw_data_dir}")

    logger.info(f"📁 Found {len(parquet_files)} parquet files")

    if test_mode:
        parquet_files = parquet_files[:5]  # Use 5 files for test
        logger.info(f"🧪 Test mode: processing {len(parquet_files)} files")

    # Read all files
    dfs = []
    successful_files = 0

    for file_path in tqdm(parquet_files, desc="Reading files"):
        try:
            # Read with required columns (adapted to actual schema)
            df = pl.read_parquet(
                file_path,
                columns=[
                    "ticker",
                    "underlying",
                    "window_start",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )

            if df.height > 0:
                dfs.append(df)
                successful_files += 1

        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")

    logger.info(f"✅ Successfully read {successful_files} files")

    if not dfs:
        raise ValueError("No data could be read from any files")

    # Combine all data
    logger.info("🔗 Combining all minute data...")
    combined_df = pl.concat(dfs, how="vertical_relaxed")
    logger.info(f"📊 Combined dataset: {combined_df.shape}")

    # Parse option codes
    logger.info("🔍 Parsing option codes...")

    # Filter to universe first (using underlying column)
    combined_df = combined_df.filter(pl.col("underlying").is_in(UNIVERSE))
    logger.info(f"📊 After universe filter: {combined_df.shape}")

    # Vectorized parsing - much faster than row-by-row
    parsed_df = parse_option_codes_vectorized(combined_df)
    logger.info(f"📊 After parsing: {parsed_df.shape}")

    # Convert data types and add timestamp
    logger.info("🔄 Converting data types...")
    parsed_df = parsed_df.with_columns(
        [
            # Convert nanosecond timestamp string to datetime
            pl.col("window_start")
            .cast(pl.Int64)
            .map_elements(
                lambda x: datetime.fromtimestamp(x / 1_000_000_000),
                return_dtype=pl.Datetime,
            )
            .alias("timestamp"),
            pl.col("exp_date").str.to_date("%Y%m%d"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.col("strike").cast(pl.Float64),
        ]
    )

    return parsed_df


def apply_filters(df: pl.DataFrame) -> pl.DataFrame:
    """Apply universe, date range, market hours, and holiday filters."""
    logger.info("🔍 Applying filters...")

    # Universe filter (already applied during parsing)
    logger.info(f"📊 Universe tickers: {df['underlying'].n_unique()}")

    # Date range filter
    date_start = pl.datetime(2021, 7, 7)
    date_end = pl.datetime(2024, 7, 7)

    df = df.filter(
        (pl.col("timestamp") >= date_start) & (pl.col("timestamp") <= date_end)
    )
    logger.info(f"📅 After date filter: {df.shape}")

    # Market hours filter
    df = filter_market_hours(df)
    logger.info(f"🕐 After market hours filter: {df.shape}")

    # Holiday filter
    df = filter_holidays(df)
    logger.info(f"🎃 After holiday filter: {df.shape}")

    # Add date column
    df = df.with_columns([pl.col("timestamp").dt.date().alias("date")])

    return df


def aggregate_to_30min(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate 1-minute data to 30-minute bars."""
    logger.info("📊 Aggregating to 30-minute bars...")

    # Create 30-minute floor timestamp - only keep :00 and :30 minutes
    df = df.with_columns([pl.col("timestamp").dt.truncate("30m").alias("bar_time")])

    # Group and aggregate to create OHLCV bars
    df_30min = df.group_by(
        ["underlying", "exp_date", "option_type", "strike", "bar_time", "date"]
    ).agg(
        [
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        ]
    )

    # Rename columns for consistency
    df_30min = df_30min.with_columns(
        [
            pl.col("underlying").alias("ticker"),
            pl.col("bar_time").alias("timestamp"),
            pl.col("bar_time").alias("window_start"),
        ]
    ).drop("bar_time")

    logger.info(f"📊 30-min bars created: {df_30min.shape}")

    # Calculate bars per day to validate
    bars_per_day = (
        df_30min.group_by(["ticker", "date"])
        .agg(pl.count().alias("bars"))
        .select("bars")
        .mean()
        .item()
    )

    logger.info(f"📊 Average bars per day: {bars_per_day:.1f} (target: ~13)")

    return df_30min


def add_underlying_features(df_30min: pl.DataFrame) -> pl.DataFrame:
    """Add simple features at underlying level."""
    logger.info("📈 Adding underlying-level features...")

    # Load stocks data - try 30-min first, fallback to daily
    stock_paths = [
        "staged/bronze_stocks_30min.parquet",
        "staged/bronze_stocks_daily_combined.parquet",
    ]

    stock_df = None
    for path in stock_paths:
        if Path(path).exists():
            logger.info(f"📂 Loading stock data from {path}")
            stock_df = pl.read_parquet(path)
            break

    if stock_df is None:
        logger.warning("⚠️ No stock history found - skipping features")
        return df_30min

    # Rename ticker to underlying for consistency
    if "ticker" in stock_df.columns:
        stock_df = stock_df.rename({"ticker": "underlying"})

    # Create features based on available data
    if "staged/bronze_stocks_daily_combined.parquet" in str(path):
        # Daily stock data - create daily features
        logger.info("🔄 Computing daily underlying features...")

        stock_df = stock_df.sort(["underlying", "date"])

        stock_features = (
            stock_df.group_by("underlying")
            .agg(
                [
                    pl.col("date"),
                    pl.col("close"),
                    pl.col("close").pct_change().alias("pct_ret_1d"),
                    (pl.col("close") / pl.col("close").shift(1))
                    .log()
                    .alias("log_ret_1d"),
                    pl.col("close")
                    .pct_change()
                    .rolling_std(window_size=20)
                    .alias("vol_20d"),
                ]
            )
            .explode(["date", "close", "pct_ret_1d", "log_ret_1d", "vol_20d"])
        )

        # Merge onto 30-min data by ticker and date
        df_with_features = df_30min.join(
            stock_features.select(
                ["underlying", "date", "pct_ret_1d", "log_ret_1d", "vol_20d"]
            ),
            left_on=["ticker", "date"],
            right_on=["underlying", "date"],
            how="left",
        )

        logger.info(f"📊 After feature merge: {df_with_features.shape}")

    else:
        # 30-min stock data available
        logger.info("🔄 Computing 30-min underlying features...")

        # Sort and compute 30-min features
        stock_df = stock_df.sort(["underlying", "timestamp"])

        stock_features = (
            stock_df.group_by("underlying")
            .agg(
                [
                    pl.col("timestamp"),
                    pl.col("close"),
                    pl.col("close").pct_change().alias("pct_ret_30m"),
                    (pl.col("close") / pl.col("close").shift(1))
                    .log()
                    .alias("log_ret_30m"),
                    pl.col("close")
                    .pct_change()
                    .rolling_std(window_size=20)
                    .alias("vol_20b"),
                ]
            )
            .explode(["timestamp", "close", "pct_ret_30m", "log_ret_30m", "vol_20b"])
        )

        # Merge onto 30-min data by ticker and timestamp
        df_with_features = df_30min.join(
            stock_features.select(
                ["underlying", "timestamp", "pct_ret_30m", "log_ret_30m", "vol_20b"]
            ),
            left_on=["ticker", "timestamp"],
            right_on=["underlying", "timestamp"],
            how="left",
        )

        logger.info(f"📊 After feature merge: {df_with_features.shape}")

    return df_with_features


def validate_options_30min_schema(df: pl.DataFrame) -> bool:
    """Validate the final 30-min options schema."""
    logger.info("🔍 Validating options 30-min schema...")

    required_columns = [
        "ticker",
        "exp_date",
        "option_type",
        "strike",
        "timestamp",
        "window_start",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "date",
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"❌ Missing required columns: {missing_cols}")
        return False

    # Check universe coverage
    unique_tickers = df["ticker"].unique().to_list()
    expected_tickers = len(UNIVERSE)
    actual_tickers = len(unique_tickers)

    logger.info(f"📊 Universe coverage: {actual_tickers}/{expected_tickers} tickers")
    if actual_tickers < expected_tickers * 0.8:  # Allow 80% coverage
        logger.warning(f"⚠️ Low universe coverage: {unique_tickers}")

    # Check date range
    min_date = df["date"].min()
    max_date = df["date"].max()

    logger.info(f"📅 Date range: {min_date} to {max_date}")

    # Check bars per day (should be ~13: 9:30, 10:00, ..., 16:00)
    bars_per_day = (
        df.group_by(["ticker", "date"])
        .agg(pl.count().alias("bars"))
        .select("bars")
        .mean()
        .item()
    )

    logger.info(f"📊 Average bars per day: {bars_per_day:.1f} (expected: ~13)")

    if bars_per_day < 10 or bars_per_day > 15:
        logger.warning(f"⚠️ Unexpected bars per day: {bars_per_day}")

    # Verify no stray 1-min timestamps
    minute_check = df.with_columns(
        pl.col("timestamp").dt.minute().alias("minute")
    ).filter(~pl.col("minute").is_in([0, 30]))

    if minute_check.height > 0:
        logger.error(f"❌ Found {minute_check.height} non-30min timestamps")
        return False

    logger.info("✅ Schema validation PASSED")
    return True


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Bronze 30-Minute Options ETL")
    parser.add_argument(
        "--raw-data-dir", required=True, help="Raw minute data directory"
    )
    parser.add_argument(
        "--out-path", default="staged/bronze_options_30min.parquet", help="Output path"
    )
    parser.add_argument(
        "--test-mode", action="store_true", help="Test mode (process fewer files)"
    )
    parser.add_argument(
        "--workers", type=int, default=24, help="Number of parallel workers"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    start_time = time.time()
    mem_start = get_memory_usage()

    logger.info("=" * 80)
    logger.info("🚀 BRONZE OPTIONS 30-MINUTE BUILD")
    logger.info("=" * 80)
    logger.info(
        f"[STARTUP] Memory: {mem_start[0]:.1f}% used ({mem_start[1]:.1f}GB / {mem_start[2]:.1f}GB)"
    )

    try:
        # 1. Read and parse raw 1-min data
        df = read_and_parse_minute_data(args.raw_data_dir, args.test_mode)

        # 2. Apply core filters
        df = apply_filters(df)

        # 3. Aggregate to 30-min bars
        df_30min = aggregate_to_30min(df)

        # 4. Add underlying features
        df_30min = add_underlying_features(df_30min)

        # 5. Schema validation
        if not validate_options_30min_schema(df_30min):
            logger.error("❌ Schema validation failed")
            return 1

        # Test mode: show summary and exit
        if args.test_mode:
            logger.info("🧪 Test mode complete - skipping file write")
            logger.info(f"📊 Final dataset shape: {df_30min.shape}")

            # Print summary
            summary = {
                "rows": int(df_30min.height),
                "unique_tickers": int(df_30min["ticker"].n_unique()),
                "date_range": {
                    "min": str(df_30min["date"].min()),
                    "max": str(df_30min["date"].max()),
                },
                "bars_per_day": float(
                    df_30min.group_by(["ticker", "date"])
                    .agg(pl.count())
                    .select("count")
                    .mean()
                    .item()
                ),
            }
            logger.info(f"📊 Summary: {summary}")
            print(json.dumps(summary, indent=2))
            return 0

        # 6. Write output
        logger.info(f"💾 Saving to: {args.out_path}")

        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write with optimal settings
        df_30min.write_parquet(args.out_path, compression="snappy", use_pyarrow=True)

        file_size = out_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Dataset saved: {df_30min.height:,} rows, {file_size:.1f} MB")

        # Final summary
        elapsed = time.time() - start_time
        mem_end = get_memory_usage()

        summary = {
            "rows": int(df_30min.height),
            "unique_tickers": int(df_30min["ticker"].n_unique()),
            "date_range": {
                "min": str(df_30min["date"].min()),
                "max": str(df_30min["date"].max()),
            },
            "bars_per_day": float(
                df_30min.group_by(["ticker", "date"])
                .agg(pl.count())
                .select("count")
                .mean()
                .item()
            ),
        }

        logger.info("=" * 80)
        logger.info("🎉 BRONZE OPTIONS 30-MINUTE BUILD COMPLETED")
        logger.info("=" * 80)
        logger.info(f"⏱️ Total processing time: {elapsed:.1f} seconds")
        logger.info(f"📊 Final dataset: {df_30min.height:,} rows")
        logger.info(f"💾 Output: {args.out_path}")
        logger.info(f"📅 Date coverage: {df_30min['date'].n_unique()} unique dates")
        logger.info(f"🎯 Universe coverage: {df_30min['ticker'].n_unique()} tickers")
        logger.info(f"📊 Build Summary: {summary}")
        logger.info(
            f"[COMPLETION] Memory: {mem_end[0]:.1f}% used ({mem_end[1]:.1f}GB / {mem_end[2]:.1f}GB)"
        )

        print(json.dumps(summary, indent=2))
        return 0

    except Exception as e:
        logger.error(f"❌ Build failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
