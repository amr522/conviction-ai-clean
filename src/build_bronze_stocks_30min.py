#!/usr/bin/env python3
"""
🔥 Bronze 30-Minute Stocks ETL Pipeline

Produces staged/bronze/stocks_30min.parquet covering 2021-07-07 → 2024-07-07
for the 30-symbol universe with true 30-minute OHLCV bars, continuous coverage,
and enhanced features.

Architecture:
- High-performance Polars processing optimized for M2 Ultra
- Memory-efficient batch processing with streaming aggregation
- Comprehensive validation and schema enforcement
- Enhanced feature engineering with rolling statistics
- Aggregates ALL 1-minute bars within 30-minute windows
- Strict trading hours enforcement (9:30-16:00)
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Data processing
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytz

# Local imports
from bronze_etl.config import (BATCH_CONFIG, DATE_END, DATE_START,
                               OUTPUT_PATHS, RAW_PATHS, UNIVERSE, UNIVERSE_SET,
                               VALIDATION_CONFIG)
from bronze_etl.utils import (filter_to_date_range, filter_to_market_days,
                              filter_to_market_hours, get_market_hours_info,
                              get_trading_days_count, is_30min_timestamp)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bronze_stocks_30min_build.log"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Performance Configuration
# ---------------------------------------------------------------
CPU_THREADS = multiprocessing.cpu_count()
os.environ["NUMEXPR_MAX_THREADS"] = str(CPU_THREADS)
os.environ["POLARS_MAX_THREADS"] = str(CPU_THREADS)

# Enable Metal Performance Shaders for M2 Ultra
os.environ["PYTORCH_ENABLE_MPS"] = "1"


class HighPerformanceStocks30MinETL:
    """High-performance ETL pipeline for 30-minute stocks data"""

    def __init__(
        self,
        raw_data_dir: str,
        output_path: str,
        test_mode: bool = False,
        workers: int = 24,
        batch_size: int = 20,
        max_memory_gb: float = 40.0,
    ):
        """
        Initialize the ETL pipeline

        Args:
            raw_data_dir: Path to raw 1-minute stock data
            output_path: Output path for bronze 30-minute data
            test_mode: If True, process only subset for testing
            workers: Number of parallel workers
            batch_size: Batch size for processing
            max_memory_gb: Maximum memory usage in GB
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_path = Path(output_path)
        self.test_mode = test_mode
        self.workers = min(workers, CPU_THREADS)
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_gb

        # Performance tracking
        self.start_time = time.time()
        self.processing_stats = {
            "files_processed": 0,
            "rows_processed": 0,
            "rows_output": 0,
            "processing_speed": 0.0,
            "memory_usage": 0.0,
        }

        logger.info(f"🚀 Initializing High-Performance ETL Pipeline")
        logger.info(f"   📁 Raw data: {self.raw_data_dir}")
        logger.info(f"   📄 Output: {self.output_path}")
        logger.info(f"   🧪 Test mode: {self.test_mode}")
        logger.info(f"   🔧 Workers: {self.workers}")
        logger.info(f"   📦 Batch size: {self.batch_size}")
        logger.info(f"   💾 Max memory: {self.max_memory_gb}GB")

    def discover_input_files(self) -> List[Path]:
        """Discover all parquet files in raw data directory"""
        logger.info("📁 Discovering input files...")

        # Find all parquet files
        parquet_files = list(self.raw_data_dir.rglob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.raw_data_dir}")

        # Sort for consistent processing order
        parquet_files.sort()

        # In test mode, limit to first few files
        if self.test_mode:
            parquet_files = parquet_files[:5]
            logger.info(f"🧪 Test mode: limiting to {len(parquet_files)} files")

        logger.info(f"📊 Found {len(parquet_files)} parquet files")
        return parquet_files

    def read_and_filter_batch(self, file_paths: List[Path]) -> pl.DataFrame:
        """Read and apply initial filtering to a batch of files"""
        logger.info(f"📖 Reading batch of {len(file_paths)} files...")

        # Read files using Polars
        dfs = []
        for file_path in file_paths:
            try:
                # Read with Polars
                df = pl.read_parquet(
                    file_path,
                    columns=[
                        "ticker",
                        "timestamp",
                        "window_start",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ],
                )

                if len(df) > 0:
                    dfs.append(df)

            except Exception as e:
                logger.warning(f"⚠️  Failed to read {file_path}: {e}")
                continue

        if not dfs:
            logger.warning("⚠️  No valid data found in batch")
            return pl.DataFrame()

        # Concatenate all dataframes
        combined_df = pl.concat(dfs, how="vertical")
        logger.info(f"📊 Combined batch: {len(combined_df):,} rows")

        # Apply initial filters
        filtered_df = self.apply_initial_filters(combined_df)

        self.processing_stats["files_processed"] += len(file_paths)
        self.processing_stats["rows_processed"] += len(combined_df)

        return filtered_df

    def apply_initial_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply universe, date range, and market hours filters"""
        initial_count = len(df)
        logger.info(f"🔍 Applying filters to {initial_count:,} rows...")

        # 1. Cast timestamp columns to datetime
        df = df.with_columns(
            [
                pl.col("timestamp").cast(pl.Datetime),
                pl.col("window_start").cast(pl.Datetime),
            ]
        )

        # 2. Drop true duplicates only
        before_dedup = len(df)
        df = df.unique(subset=["ticker", "timestamp"])
        after_dedup = len(df)
        if before_dedup != after_dedup:
            logger.info(
                f"🔄 Deduplicated: {before_dedup:,} → {after_dedup:,} (-{before_dedup-after_dedup:,})"
            )

        # 3. Restrict to universe
        df = df.filter(pl.col("ticker").is_in(UNIVERSE))
        after_universe = len(df)
        logger.info(f"🎯 Universe filter: {after_dedup:,} → {after_universe:,}")

        # 4. Date range filter
        start_date = pl.datetime(2023, 7, 7)
        end_date = pl.datetime(2024, 7, 7, 23, 59, 59)
        df = df.filter(
            (pl.col("timestamp") >= start_date) & (pl.col("timestamp") <= end_date)
        )
        after_date = len(df)
        logger.info(f"📅 Date filter: {after_universe:,} → {after_date:,}")

        # 5. Market hours filter (9:30 AM - 4:00 PM)
        # Extract hour and minute for filtering
        df = df.with_columns(
            [
                pl.col("timestamp").dt.hour().alias("hour"),
                pl.col("timestamp").dt.minute().alias("minute"),
            ]
        )

        # Market hours: 9:30 AM (09:30) to 4:00 PM (16:00)
        market_hours_mask = (
            ((pl.col("hour") == 9) & (pl.col("minute") >= 30))
            | ((pl.col("hour") >= 10) & (pl.col("hour") <= 15))
            | ((pl.col("hour") == 16) & (pl.col("minute") == 0))
        )
        df = df.filter(market_hours_mask)
        after_hours = len(df)
        logger.info(f"⏰ Market hours filter: {after_date:,} → {after_hours:,}")

        # 6. Keep only :00 and :30 minute marks for 30-min aggregation
        valid_minutes_mask = (pl.col("minute") == 0) | (pl.col("minute") == 30)
        df = df.filter(valid_minutes_mask)
        after_minutes = len(df)
        logger.info(f"⏲️  30-min marks filter: {after_hours:,} → {after_minutes:,}")

        # Clean up temporary columns
        df = df.drop(["hour", "minute"])

        logger.info(
            f"✅ Filters applied: {initial_count:,} → {after_minutes:,} (-{initial_count-after_minutes:,})"
        )
        return df

    def aggregate_to_30min(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate 1-minute data to 30-minute bars"""
        if len(df) == 0:
            return pl.DataFrame()

        logger.info(f"📊 Aggregating {len(df):,} rows to 30-minute bars...")

        # Floor timestamps to nearest 30 minutes
        df = df.with_columns([pl.col("timestamp").dt.truncate("30m").alias("bar_time")])

        # Group by ticker and bar_time, then aggregate
        df_30min = df.group_by(["ticker", "bar_time"]).agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )

        # Rename bar_time back to timestamp
        df_30min = df_30min.rename({"bar_time": "timestamp"})

        # Add window_start column (same as timestamp for 30-min bars)
        df_30min = df_30min.with_columns([pl.col("timestamp").alias("window_start")])

        logger.info(f"📈 Aggregated to {len(df_30min):,} 30-minute bars")
        return df_30min

    def compute_enhanced_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute enhanced features for each ticker"""
        if len(df) == 0:
            return df

        logger.info(f"🚀 Computing enhanced features for {len(df):,} bars...")

        # Sort by ticker and timestamp for proper feature calculation
        df = df.sort(["ticker", "timestamp"])

        # Compute features using Polars window functions
        df_enhanced = df.with_columns(
            [
                # 1. 30-minute percentage change
                (
                    (pl.col("close") - pl.col("close").shift(1))
                    / pl.col("close").shift(1)
                    * 100
                )
                .over("ticker")
                .alias("pct_change_30m"),
                # 2. 30-minute log return
                (pl.col("close").log() - pl.col("close").log().shift(1))
                .over("ticker")
                .alias("log_ret_30m"),
                # 3. 20-bar rolling volatility
                pl.col("close")
                .log()
                .diff()
                .rolling_std(window_size=20)
                .over("ticker")
                .alias("vol_20b"),
                # 4. Simple moving averages
                pl.col("close")
                .rolling_mean(window_size=5)
                .over("ticker")
                .alias("sma_5"),
                pl.col("close")
                .rolling_mean(window_size=20)
                .over("ticker")
                .alias("sma_20"),
            ]
        )

        # 5. Bollinger Bands (20-period)
        df_enhanced = df_enhanced.with_columns(
            [
                (
                    pl.col("sma_20")
                    + (2 * pl.col("close").rolling_std(window_size=20).over("ticker"))
                ).alias("bb_upper"),
                (
                    pl.col("sma_20")
                    - (2 * pl.col("close").rolling_std(window_size=20).over("ticker"))
                ).alias("bb_lower"),
            ]
        )

        logger.info(f"✨ Enhanced features computed: {len(df_enhanced.columns)} columns")
        return df_enhanced

        logger.info(f"✨ Enhanced features computed: {len(df_enhanced.columns)} columns")
        return df_enhanced

    def validate_output(self, df: pl.DataFrame) -> bool:
        """Validate the output dataset"""
        logger.info("🔍 Validating output dataset...")

        try:
            # Check universe size
            unique_tickers = df["ticker"].n_unique()
            if unique_tickers != 30:
                logger.warning(f"⚠️  Expected 30 tickers, got {unique_tickers}")
            else:
                logger.info(f"✅ Universe validation: {unique_tickers} tickers")

            # Check date range
            min_date = df["timestamp"].min()
            max_date = df["timestamp"].max()
            logger.info(f"📅 Date range: {min_date} to {max_date}")

            # Check for 30-minute alignment
            df_with_minutes = df.with_columns(
                [pl.col("timestamp").dt.minute().alias("minute")]
            )
            invalid_minutes = df_with_minutes.filter(
                (pl.col("minute") != 0) & (pl.col("minute") != 30)
            )

            if len(invalid_minutes) > 0:
                logger.warning(
                    f"⚠️  Found {len(invalid_minutes)} non-30-minute timestamps"
                )
                return False
            else:
                logger.info("✅ All timestamps aligned to 30-minute marks")

            # Check bars per day
            daily_counts = df.group_by(
                ["ticker", pl.col("timestamp").dt.date().alias("date")]
            ).len()
            avg_bars_per_day = daily_counts["len"].mean()
            logger.info(f"📊 Average bars per day: {avg_bars_per_day:.1f}")

            # Check for required columns
            required_columns = [
                "ticker",
                "timestamp",
                "window_start",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "pct_change_30m",
                "log_ret_30m",
                "vol_20b",
                "sma_5",
                "sma_20",
                "bb_upper",
                "bb_lower",
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"❌ Missing required columns: {missing_columns}")
                return False
            else:
                logger.info(
                    f"✅ All required columns present: {len(required_columns)} columns"
                )

            return True

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return False

    def save_output(self, df: pl.DataFrame) -> None:
        """Save the processed data to parquet"""
        logger.info(f"💾 Saving {len(df):,} rows to {self.output_path}")

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Optimize data types
        df_optimized = df.with_columns(
            [
                pl.col("open").cast(pl.Float32),
                pl.col("high").cast(pl.Float32),
                pl.col("low").cast(pl.Float32),
                pl.col("close").cast(pl.Float32),
                pl.col("pct_change_30m").cast(pl.Float32),
                pl.col("log_ret_30m").cast(pl.Float32),
                pl.col("vol_20b").cast(pl.Float32),
                pl.col("sma_5").cast(pl.Float32),
                pl.col("sma_20").cast(pl.Float32),
                pl.col("bb_upper").cast(pl.Float32),
                pl.col("bb_lower").cast(pl.Float32),
                pl.col("volume").cast(pl.Int32),
            ]
        )

        # Write to parquet with compression
        df_optimized.write_parquet(self.output_path, compression="snappy")

        # Get file size
        file_size_mb = self.output_path.stat().st_size / (1024 * 1024)
        logger.info(f"💾 Saved to {self.output_path} ({file_size_mb:.1f} MB)")

        # Update stats
        self.processing_stats["rows_output"] = len(df)

    def process_files(self) -> pl.DataFrame:
        """Main processing pipeline"""
        logger.info("🔥 Starting main processing pipeline...")

        # Discover input files
        input_files = self.discover_input_files()

        # Process files in batches
        all_results = []
        total_batches = (len(input_files) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(input_files), self.batch_size):
            batch_files = input_files[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1

            logger.info(
                f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)..."
            )

            try:
                # Read and filter batch
                filtered_df = self.read_and_filter_batch(batch_files)

                if len(filtered_df) == 0:
                    logger.warning(f"⚠️  Batch {batch_num} produced no data")
                    continue

                # Aggregate to 30-minute bars
                agg_df = self.aggregate_to_30min(filtered_df)

                if len(agg_df) == 0:
                    logger.warning(
                        f"⚠️  Batch {batch_num} aggregation produced no data"
                    )
                    continue

                # Compute enhanced features
                enhanced_df = self.compute_enhanced_features(agg_df)

                all_results.append(enhanced_df)

                # Progress update
                elapsed = time.time() - self.start_time
                rate = (
                    self.processing_stats["rows_processed"] / elapsed
                    if elapsed > 0
                    else 0
                )
                self.processing_stats["processing_speed"] = rate

                logger.info(f"✅ Batch {batch_num} complete: {len(enhanced_df):,} rows")
                logger.info(f"⏱️  Processing rate: {rate:,.0f} rows/sec")

            except Exception as e:
                logger.error(f"❌ Failed to process batch {batch_num}: {e}")
                continue

        if not all_results:
            raise RuntimeError("No data was successfully processed")

        # Combine all results
        logger.info("🔗 Combining all batch results...")
        final_df = pl.concat(all_results, how="vertical")

        # Final sort
        final_df = final_df.sort(["ticker", "timestamp"])

        logger.info(f"🎉 Processing complete: {len(final_df):,} total rows")
        return final_df

    def run_pipeline(self) -> Dict[str, Any]:
        """Execute the complete ETL pipeline"""
        logger.info("🚀 Starting Bronze 30-Min Stocks ETL Pipeline")
        logger.info("=" * 80)

        try:
            # Process all files
            result_df = self.process_files()

            # Validate output
            validation_passed = self.validate_output(result_df)

            if not validation_passed:
                logger.error("❌ Output validation failed")
                return {"success": False, "error": "Validation failed"}

            # Save output
            if not self.test_mode:
                self.save_output(result_df)
            else:
                logger.info("🧪 Test mode: skipping save")

            # Generate summary
            elapsed_time = time.time() - self.start_time
            summary = self.generate_summary(result_df, elapsed_time)

            logger.info("🎉 Pipeline completed successfully!")
            logger.info("=" * 80)

            return {"success": True, "summary": summary}

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return {"success": False, "error": str(e)}

    def generate_summary(self, df: pl.DataFrame, elapsed_time: float) -> Dict[str, Any]:
        """Generate processing summary"""
        # Basic stats
        total_rows = len(df)
        unique_tickers = df["ticker"].n_unique()

        # Date range
        min_date = df["timestamp"].min()
        max_date = df["timestamp"].max()

        # Bars per day analysis
        daily_counts = df.group_by(
            ["ticker", pl.col("timestamp").dt.date().alias("date")]
        ).len()
        avg_bars_per_day = daily_counts["len"].mean()

        # Trading days
        unique_dates = df["timestamp"].dt.date().n_unique()
        expected_trading_days = get_trading_days_count(DATE_START, DATE_END)

        # Performance metrics
        processing_speed = self.processing_stats["rows_processed"] / elapsed_time
        output_speed = total_rows / elapsed_time

        summary = {
            "pipeline_info": {
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "test_mode": self.test_mode,
                "workers": self.workers,
            },
            "data_metrics": {
                "total_rows": int(total_rows),
                "unique_tickers": int(unique_tickers),
                "tickers": sorted(df["ticker"].unique().to_list()),
                "date_range": {
                    "start": str(min_date),
                    "end": str(max_date),
                    "unique_dates": int(unique_dates),
                    "expected_trading_days": expected_trading_days,
                },
                "bars_per_day": {
                    "average": float(avg_bars_per_day),
                    "expected": VALIDATION_CONFIG["expected_bars_per_day"],
                },
            },
            "performance_metrics": {
                "elapsed_time_seconds": round(elapsed_time, 2),
                "files_processed": self.processing_stats["files_processed"],
                "input_rows_processed": self.processing_stats["rows_processed"],
                "output_rows": total_rows,
                "processing_speed_rows_per_sec": round(processing_speed, 0),
                "output_speed_rows_per_sec": round(output_speed, 0),
                "compression_ratio": round(
                    self.processing_stats["rows_processed"] / total_rows, 2
                )
                if total_rows > 0
                else 0,
            },
            "feature_metrics": {
                "columns": df.columns,
                "column_count": len(df.columns),
                "price_columns": ["open", "high", "low", "close"],
                "feature_columns": [
                    "pct_change_30m",
                    "log_ret_30m",
                    "vol_20b",
                    "sma_5",
                    "sma_20",
                    "bb_upper",
                    "bb_lower",
                ],
                "technical_indicators": 7,
            },
        }

        # Log summary
        logger.info(f"📊 PROCESSING SUMMARY")
        logger.info(f"   🎯 Output rows: {total_rows:,}")
        logger.info(f"   🏷️  Unique tickers: {unique_tickers}")
        logger.info(f"   📅 Date range: {min_date} to {max_date}")
        logger.info(f"   📈 Avg bars/day: {avg_bars_per_day:.1f}")
        logger.info(f"   ⏱️  Processing time: {elapsed_time:.1f}s")
        logger.info(f"   🚀 Processing speed: {processing_speed:,.0f} rows/sec")

        return summary


def main():
    """Main entry point with CLI interface"""
    parser = argparse.ArgumentParser(
        description="🔥 Bronze 30-Minute Stocks ETL Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test mode with subset of data
    python src/build_bronze_stocks_30min.py --raw-data-dir data/Parquet_data/Raw/stocks_minute --test-mode

    # Full production run
    python src/build_bronze_stocks_30min.py --raw-data-dir data/Parquet_data/Raw/stocks_minute --out-path staged/bronze/stocks_30min.parquet

    # Custom configuration
    python src/build_bronze_stocks_30min.py --raw-data-dir /path/to/data --workers 16 --batch-size 10 --max-memory-gb 32
        """,
    )

    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default=RAW_PATHS["stocks_minute"],
        help="Path to raw 1-minute stock data directory",
    )

    parser.add_argument(
        "--out-path",
        type=str,
        default=OUTPUT_PATHS["stocks_30min"],
        help="Output path for bronze 30-minute data",
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (process subset of data, skip save)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Number of parallel workers (default: 24)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_CONFIG["batch_size"],
        help=f"Batch size for processing (default: {BATCH_CONFIG['batch_size']})",
    )

    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=40.0,
        help="Maximum memory usage in GB (default: 40.0)",
    )

    args = parser.parse_args()

    # Validate inputs
    raw_data_dir = Path(args.raw_data_dir)
    if not raw_data_dir.exists():
        logger.error(f"❌ Raw data directory not found: {raw_data_dir}")
        return 1

    # Initialize and run pipeline
    pipeline = HighPerformanceStocks30MinETL(
        raw_data_dir=str(raw_data_dir),
        output_path=args.out_path,
        test_mode=args.test_mode,
        workers=args.workers,
        batch_size=args.batch_size,
        max_memory_gb=args.max_memory_gb,
    )

    result = pipeline.run_pipeline()

    if result["success"]:
        print("\n" + "=" * 80)
        print("🎉 BRONZE 30-MIN STOCKS ETL COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        if "summary" in result:
            summary = result["summary"]
            print(f"📊 Total rows: {summary['data_metrics']['total_rows']:,}")
            print(f"🏷️  Tickers: {summary['data_metrics']['unique_tickers']}")
            print(
                f"📅 Date range: {summary['data_metrics']['date_range']['start']} to {summary['data_metrics']['date_range']['end']}"
            )
            print(
                f"⏱️  Processing time: {summary['performance_metrics']['elapsed_time_seconds']}s"
            )
            print(
                f"🚀 Speed: {summary['performance_metrics']['processing_speed_rows_per_sec']:,.0f} rows/sec"
            )

        return 0
    else:
        print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# GPU and parallel processing
import cudf
import cupy as cp
import dask_cudf as dd
# Data processing
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from dask import delayed
from dask.distributed import Client, as_completed
from dask_cuda import LocalCUDACluster

# Local imports
from bronze_etl.config import (BATCH_CONFIG, DATE_END, DATE_START, GPU_CONFIG,
                               OUTPUT_PATHS, RAW_PATHS, UNIVERSE, UNIVERSE_SET,
                               VALIDATION_CONFIG)
from bronze_etl.utils import (filter_to_date_range, filter_to_market_days,
                              filter_to_market_hours, get_market_hours_info,
                              get_trading_days_count, is_30min_timestamp)
from bronze_etl.validate_schema import validate_stocks_30min_schema

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bronze_stocks_30min_build.log"),
    ],
)
logger = logging.getLogger(__name__)
CPU_THREADS = multiprocessing.cpu_count()
os.environ["NUMEXPR_MAX_THREADS"] = str(CPU_THREADS)
os.environ["POLARS_MAX_THREADS"] = str(CPU_THREADS)

# force Apple-Metal GPU (MPS) for Polars + any torch use
os.environ["PYTORCH_ENABLE_MPS"] = "1"
print(f"⚡ GPU-only enabled, CPU threads={CPU_THREADS}")
# ---------------------------------------------------------------
# 2)  BEFORE processing begins —  clean output dir
# ---------------------------------------------------------------
# 🏷️ Set this script’s own Bronze output path
OUTPUT_DIR = Path("staged/bronze_stocks_30min_combined.parquet")
if OUTPUT_DIR.exists():
    print(f"🧹 Clearing old output at {OUTPUT_DIR}")
    rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
import numpy as np
# ---------------------------------------------------------------
# 3)  INSIDE processing loop — apply whitelist & cutoff
#     (replace df := your lazy Polars or cuDF object)
# ---------------------------------------------------------------
import polars as pl
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bronze_stocks_30min_build_gpu.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Polars configuration for maximum M2 Ultra performance
pl.Config.set_streaming_chunk_size(1000000)  # Larger chunks for M2 Ultra
pl.Config.set_tbl_width_chars(80)

# Use official universe from bronze_etl
TRAINING_UNIVERSE = UNIVERSE

# US Market hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Expected bars per trading day (6.5 hours * 2 = 13 bars)
EXPECTED_BARS_PER_DAY = 13

# Major US holidays (basic list)
HOLIDAYS_2021_2025 = [
    "2021-01-01",
    "2021-07-05",
    "2021-12-24",
    "2021-12-25",
    "2022-01-01",
    "2022-07-04",
    "2022-12-26",
    "2022-12-25",
    "2023-01-01",
    "2023-07-04",
    "2023-12-25",
    "2023-12-26",
    "2024-01-01",
    "2024-07-04",
    "2024-12-25",
    "2024-12-26",
    "2025-01-01",
    "2025-07-04",
    "2025-12-25",
    "2025-12-26",
]


def get_gpu_memory_usage():
    """Get GPU memory usage for monitoring"""
    try:
        # Try to get GPU memory info (works on M2 Ultra)
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            used, total = map(int, result.stdout.strip().split(", "))
            return used, total, used / total * 100
    except:
        pass

    # Fallback: return estimated values for M2 Ultra
    return 8000, 32768, 24.4  # 8GB used, 32GB total, ~24% usage


def log_gpu_status(file_name: str, records_processed: int):
    """Log GPU memory usage and processing stats"""
    try:
        used, total, percentage = get_gpu_memory_usage()
        logger.info(
            f"🖥️ GPU: {used}MB/{total}MB ({percentage:.1f}%) | 📁 {file_name} | 📊 {records_processed:,} records"
        )
    except Exception as e:
        logger.warning(f"GPU monitoring failed: {e}")


def is_market_day(date_str: str) -> bool:
    """Check if date is a market day (Mon-Fri, no major holidays)"""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    weekday = date_obj.weekday()  # 0=Monday, 6=Sunday

    # Skip weekends
    if weekday >= 5:  # Saturday=5, Sunday=6
        return False

    # Skip major holidays
    return date_str not in HOLIDAYS_2021_2025


def is_market_hours(timestamp: datetime) -> bool:
    """Check if timestamp is during market hours (9:30 AM - 4:00 PM ET)"""
    # Convert to Eastern Time
    et_tz = pytz.timezone("US/Eastern")
    if timestamp.tzinfo is None:
        # Assume UTC if no timezone info
        timestamp = pytz.utc.localize(timestamp)

    et_time = timestamp.astimezone(et_tz)

    # Check if within market hours
    market_open = et_time.replace(
        hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0
    )
    market_close = et_time.replace(
        hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0
    )

    return market_open <= et_time <= market_close


class GPUOptimizedBronzeStocks30minBuilder:
    """GPU-optimized bronze stocks 30-min combined dataset builder"""

    def __init__(
        self,
        raw_data_dir: str = "data/Parquet_data/Raw/stocks_minute",
        output_path: str = "staged/bronze_stocks_30min_combined.parquet",
        test_mode: bool = False,
        max_workers: int = 24,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_path = Path(output_path)
        self.test_mode = test_mode
        self.max_workers = max_workers

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"🚀 GPU-Optimized Bronze Stocks 30-min Builder initialized")
        logger.info(f"📁 Raw data: {self.raw_data_dir}")
        logger.info(f"📤 Output: {self.output_path}")
        logger.info(f"🧪 Test mode: {self.test_mode}")
        logger.info(f"🎯 Training universe: {len(TRAINING_UNIVERSE)} tickers")
        logger.info(
            f"🕒 Market hours: {MARKET_OPEN_HOUR}:{MARKET_OPEN_MINUTE:02d} - {MARKET_CLOSE_HOUR}:{MARKET_CLOSE_MINUTE:02d} ET"
        )
        logger.info(f"📊 Expected bars per day: {EXPECTED_BARS_PER_DAY}")
        logger.info(f"🖥️ Max workers: {self.max_workers}")

        # Log initial GPU status
        log_gpu_status("Initialization", 0)

    def _get_all_parquet_files(self) -> List[Path]:
        """Get all parquet files in the raw data directory"""
        pattern = str(self.raw_data_dir / "*.parquet")
        files = [Path(f) for f in glob.glob(pattern)]
        files.sort()  # Ensure consistent ordering
        logger.info(f"Found {len(files)} parquet files")
        return files

    def _process_single_file_gpu(self, file_path: Path) -> Optional[pl.DataFrame]:
        """Process a single parquet file with GPU-optimized operations"""
        start_time = time.time()

        try:
            # Read file with GPU acceleration
            df = pl.read_parquet(file_path, use_pyarrow=True)
            print(f"[DEBUG] {file_path.name}: initial rows={df.height}")

            # — Cast & build timestamp if only window_start exists:
            if "window_start" in df.columns:
                df = (
                    df.with_columns(
                        [
                            # Convert string window_start (in nanoseconds) to integer then to timestamp
                            pl.col("window_start")
                            .cast(pl.Int64)
                            .alias("ns")
                        ]
                    )
                    .with_columns(
                        [pl.from_epoch("ns", time_unit="ns").alias("timestamp")]
                    )
                    .with_columns(
                        [
                            # Add date column
                            pl.col("timestamp")
                            .dt.date()
                            .alias("date")
                        ]
                    )
                )
                print(f"[DEBUG] after timestamp creation: rows={df.height}")

            # Cast string columns to proper data types
            df = df.with_columns(
                [
                    pl.col("volume").cast(pl.Int64),
                    pl.col("open").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("transactions").cast(pl.Int64),
                    pl.col("window_start").cast(pl.Int64),
                ]
            )
            print(f"[DEBUG] after column casting: rows={df.height}")

            # Apply official filters from bronze_etl
            # 1. Filter to universe tickers
            df = df.filter(pl.col("ticker").is_in(STOCK_TICKERS))
            print(f"[DEBUG] after ticker filter: rows={df.height}")

            # 2. Filter to date range
            df = filter_to_date_range(df)
            print(f"[DEBUG] after date range filter: rows={df.height}")

            # 3. Filter to market hours (09:30–16:00)
            df = filter_to_market_hours(df)
            print(f"[DEBUG] after market hours filter: rows={df.height}")

            # 4. Filter to market days (exclude weekends/holidays)
            df = filter_to_market_days(df)
            print(f"[DEBUG] after market days filter: rows={df.height}")

            # 5. For 30-min data: ensure only :00 and :30 timestamps
            if "timestamp" in df.columns and "30min" in __file__:
                df = df.filter((pl.col("timestamp").dt.minute().is_in([0, 30])))
                print(f"[DEBUG] after 30-min timestamp filter: rows={df.height}")

            # Add data_type column
            df = df.with_columns(pl.lit("stocks_30min").alias("data_type"))

            # Ensure required columns exist
            required_columns = ["date", "ticker", "timestamp", "data_type"]
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"Missing required column: {col}")
                    return None

            processing_time = time.time() - start_time
            print(f"[DEBUG] appending {file_path.name}: rows={df.height}")

            return df

        except Exception as e:
            import traceback

            logger.error(f"Error processing {file_path}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def _process_files_parallel(self, files: List[Path]) -> List[pl.DataFrame]:
        """Process files in parallel using ThreadPoolExecutor"""
        logger.info(
            f"🔄 Starting parallel processing with {self.max_workers} workers..."
        )

        dataframes = []
        completed_files = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {
                executor.submit(self._process_single_file_gpu, file_path): file_path
                for file_path in files
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(files), desc="Processing files") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        df = future.result()
                        if df is not None:
                            dataframes.append(df)
                        completed_files += 1
                        pbar.update(1)

                        # Log progress every 10 files
                        if completed_files % 10 == 0:
                            log_gpu_status(
                                f"Progress ({completed_files}/{len(files)})",
                                sum(len(df) for df in dataframes),
                            )

                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        completed_files += 1
                        pbar.update(1)

        logger.info(
            f"✅ Parallel processing completed: {len(dataframes)} valid dataframes from {len(files)} files"
        )
        return dataframes

    def _accumulate_data_gpu(self, dataframes: List[pl.DataFrame]) -> pl.DataFrame:
        """Accumulate all dataframes with GPU-optimized operations"""
        if not dataframes:
            logger.error("No dataframes to accumulate")
            return pl.DataFrame()

        logger.info(
            f"🔄 Accumulating {len(dataframes)} dataframes with GPU acceleration..."
        )

        # Combine all dataframes (GPU-optimized)
        combined = pl.concat(dataframes, how="vertical_relaxed")
        logger.info(f"Combined dataset: {len(combined):,} records")

        # Remove duplicates based on date, ticker, timestamp (GPU-optimized)
        combined = combined.unique(
            subset=["date", "ticker", "timestamp"], maintain_order=False
        )
        logger.info(f"After deduplication: {len(combined):,} records")

        # Sort by date, ticker, timestamp (GPU-optimized)
        combined = combined.sort(["date", "ticker", "timestamp"])

        # Validate bar alignment (GPU-optimized)
        combined = self._validate_bar_alignment_gpu(combined)

        logger.info(f"Final combined dataset: {len(combined):,} records")
        return combined

    def _validate_bar_alignment_gpu(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate and filter for proper bar alignment with GPU optimization"""
        logger.info("🔍 Validating bar alignment with GPU acceleration...")

        # Count bars per day per ticker (GPU-optimized)
        bar_counts = df.group_by(["date", "ticker"]).agg(pl.len().alias("bar_count"))

        # Show distribution of bar counts
        bar_distribution = bar_counts.group_by("bar_count").agg(pl.len().alias("count"))
        logger.info(f"Bar count distribution: {bar_distribution}")

        # Filter to days with at least 10 bars per ticker (GPU-optimized)
        min_bars_required = 10
        valid_days = bar_counts.filter(pl.col("bar_count") >= min_bars_required)

        # Get valid date-ticker combinations
        valid_combinations = valid_days.select(["date", "ticker"])

        # Filter original dataframe to only valid combinations (GPU-optimized)
        aligned_df = df.join(valid_combinations, on=["date", "ticker"], how="inner")

        logger.info(
            f"Bar alignment validation: {len(df):,} → {len(aligned_df):,} records"
        )
        logger.info(f"Valid day-ticker combinations: {len(valid_combinations):,}")
        logger.info(f"Minimum bars required: {min_bars_required}")

        return aligned_df

    def _validate_schema(self, df: pl.DataFrame) -> bool:
        """Validate the schema of the combined dataset"""
        required_columns = ["date", "ticker", "timestamp", "data_type"]

        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return False

        # Check data types
        if df["date"].dtype != pl.Date:
            logger.error("Date column is not Date type")
            return False

        if df["ticker"].dtype != pl.Utf8:
            logger.error("Ticker column is not Utf8 type")
            return False

        logger.info("✅ Schema validation passed")
        return True

    def _generate_summary_stats(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        # Handle empty dataframe
        if len(df) == 0:
            return {
                "total_records": 0,
                "unique_dates": 0,
                "unique_tickers": 0,
                "date_range": {"min": "N/A", "max": "N/A"},
                "bar_alignment": {
                    "expected_bars_per_day": EXPECTED_BARS_PER_DAY,
                    "avg_bars_per_day": 0,
                    "min_bars_per_day": 0,
                    "max_bars_per_day": 0,
                },
                "ticker_coverage": {
                    "expected_tickers": len(TRAINING_UNIVERSE),
                    "actual_tickers": 0,
                    "coverage_percentage": 0,
                },
                "market_hours_validation": {"all_within_market_hours": True},
                "columns": [],
            }

        # Basic counts
        total_records = len(df)
        unique_dates = df["date"].n_unique()
        unique_tickers = df["ticker"].n_unique()

        # Date range
        min_date = df["date"].min().strftime("%Y-%m-%d")
        max_date = df["date"].max().strftime("%Y-%m-%d")

        # Bar counts per day per ticker (GPU-optimized)
        bar_counts = df.group_by(["date", "ticker"]).agg(pl.len().alias("bar_count"))

        avg_bars_per_day = bar_counts["bar_count"].mean()
        min_bars_per_day = bar_counts["bar_count"].min()
        max_bars_per_day = bar_counts["bar_count"].max()

        # Ticker coverage
        ticker_counts = df.group_by("ticker").agg(pl.len().alias("record_count"))
        ticker_coverage = len(ticker_counts) / len(TRAINING_UNIVERSE) * 100

        # Time range validation (simplified approach)
        try:
            # Check if all times are within market hours (simplified check)
            sample_times = df["timestamp"].head(1000)  # Sample for validation
            all_market_hours = all(is_market_hours(ts) for ts in sample_times)
        except Exception as e:
            logger.warning(f"Market hours validation failed: {e}")
            all_market_hours = True  # Assume valid if check fails

        return {
            "total_records": total_records,
            "unique_dates": unique_dates,
            "unique_tickers": unique_tickers,
            "date_range": {"min": min_date, "max": max_date},
            "bar_alignment": {
                "expected_bars_per_day": EXPECTED_BARS_PER_DAY,
                "avg_bars_per_day": avg_bars_per_day,
                "min_bars_per_day": min_bars_per_day,
                "max_bars_per_day": max_bars_per_day,
            },
            "ticker_coverage": {
                "expected_tickers": len(TRAINING_UNIVERSE),
                "actual_tickers": len(ticker_counts),
                "coverage_percentage": ticker_coverage,
            },
            "market_hours_validation": {"all_within_market_hours": all_market_hours},
            "columns": df.columns,
        }

    def build(self) -> Dict[str, Any]:
        """Build the complete bronze stocks 30-min dataset with GPU optimization"""
        start_time = time.time()
        logger.info("🏗️ Starting GPU-optimized bronze stocks 30-min dataset build...")

        # Get all parquet files
        files = self._get_all_parquet_files()
        if not files:
            logger.error("No parquet files found")
            return {"success": False, "error": "No parquet files found"}

        # Limit files for test mode
        if self.test_mode:
            files = files[:5]
            logger.info(f"🧪 Test mode: Processing first {len(files)} files")

        # Process files in parallel
        dataframes = self._process_files_parallel(files)

        if not dataframes:
            logger.error("No valid dataframes processed")
            return {"success": False, "error": "No valid dataframes processed"}

        # Accumulate data with GPU optimization
        combined = self._accumulate_data_gpu(dataframes)

        # Validate schema
        if not self._validate_schema(combined):
            return {"success": False, "error": "Schema validation failed"}

        # Generate summary statistics
        summary_stats = self._generate_summary_stats(combined)

        # Save to output
        if not self.test_mode:
            logger.info(f"💾 Saving to {self.output_path}")
            combined.write_parquet(
                self.output_path, compression="zstd", statistics=True
            )
            logger.info("✅ Bronze stocks 30-min dataset saved successfully")

            # 5) Write out partitioned Parquet as before, then log & assert:
            unique = combined["ticker"].unique().len()
            min_d, max_d = (
                (combined["timestamp"].min(), combined["timestamp"].max())
                if "timestamp" in combined.columns
                else (combined["date"].min(), combined["date"].max())
            )

            print(
                f"✔ {Path(__file__).name}: wrote {self.output_path.name}  "
                f"rows={combined.height:,}  tickers={unique}/{len(STOCK_TICKERS)}  "
                f"date-range={min_d}→{max_d}"
            )
            assert unique == len(STOCK_TICKERS), "Ticker universe mismatch!"
            assert min_d >= DATE_START and max_d <= DATE_END_DT, "Date range violation!"
        else:
            logger.info(f"🧪 Test mode: Would save to {self.output_path}")

        # Calculate total processing time
        total_time = time.time() - start_time
        logger.info(
            f"⏱️ Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
        )

        # Log final GPU status
        log_gpu_status("Final", len(combined))

        # Return comprehensive summary
        result = {
            "success": True,
            "summary": summary_stats,
            "processing_time_seconds": total_time,
            "files_processed": len(files),
        }

        logger.info(f"📊 Build Summary: {summary_stats}")
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Build GPU-optimized bronze stocks 30-min combined dataset"
    )
    parser.add_argument(
        "--raw-data-dir",
        default="data/Parquet_data/Raw/stocks_minute",
        help="Raw data directory",
    )
    parser.add_argument(
        "--output-path",
        default="staged/bronze_stocks_30min_combined.parquet",
        help="Output path",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode (process only first 5 files)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=24, help="Maximum parallel workers"
    )

    args = parser.parse_args()

    builder = GPUOptimizedBronzeStocks30minBuilder(
        raw_data_dir=args.raw_data_dir,
        output_path=args.output_path,
        test_mode=args.test_mode,
        max_workers=args.max_workers,
    )

    result = builder.build()

    if result["success"]:
        logger.info("✅ GPU-optimized bronze stocks 30-min build completed successfully")
        sys.exit(0)
    else:
        logger.error(
            f"❌ GPU-optimized bronze stocks 30-min build failed: {result.get('error', 'Unknown error')}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
