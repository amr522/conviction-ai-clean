import argparse
import os
from datetime import timedelta
from pathlib import Path

import polars as pl
from prefect import task
from prefect.tasks import task_input_hash

from utils.performance_utils import PERF_CONFIG, optimize_join_performance
from utils.profiling import profile_memory_and_time, profile_time


@task(
    name="build_intraday_dataset",
    description="Build intraday dataset by joining 30-min stocks and options data",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    retries=3,
    retry_delay_seconds=60,
)
@profile_memory_and_time
def run(date: str, dry_run: bool = False) -> dict:
    """
    Build intraday dataset by joining 30-minute stocks and options data.

    Args:
        date: The processing date (for logging/tracking)
        dry_run: If True, simulate processing without writing files

    Returns:
        dict: Status information about the processing
    """
    try:
        print(f"Starting intraday dataset build for date: {date}")

        # Input/output paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stocks_path = os.path.join(project_root, "staged/stocks_30min_clean.parquet")
        options_path = os.path.join(project_root, "staged/options_30min_clean.parquet")
        output_dir = os.path.join(project_root, "staged")
        output_path = os.path.join(output_dir, "intraday_master.parquet")

        print(f"Stocks path: {stocks_path}")
        print(f"Options path: {options_path}")
        print(f"Output path: {output_path}")

        if not dry_run:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Validate input files exist
        if not os.path.exists(stocks_path):
            raise FileNotFoundError(f"Stocks data file not found: {stocks_path}")
        if not os.path.exists(options_path):
            raise FileNotFoundError(f"Options data file not found: {options_path}")

        # Load and validate stocks data (larger table)
        stocks_df = pl.scan_parquet(stocks_path, extra_columns="ignore")
        stocks_data = stocks_df.collect()
        if stocks_data.shape[0] == 0:
            raise ValueError("Stocks data is empty")
        print(f"\nStocks shape: {stocks_data.shape}")

        required_stock_cols = ["timestamp", "ticker", "stock30_close", "stock30_volume"]
        missing_cols = [
            col for col in required_stock_cols if col not in stocks_data.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing required columns in stocks data: {missing_cols}")

        # Load and validate options data (smaller table for broadcast)
        options_df = pl.scan_parquet(options_path, extra_columns="ignore")
        options_data = options_df.collect()
        if options_data.shape[0] == 0:
            raise ValueError("Options data is empty")
        print(f"Options shape: {options_data.shape}")

        required_option_cols = ["timestamp", "ticker", "opt30_strike", "opt30_type"]
        missing_cols = [
            col for col in required_option_cols if col not in options_data.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in options data: {missing_cols}"
            )

        # Use optimized join function with profiling
        df = optimize_join_performance(
            stocks_df, options_df, on=["timestamp", "ticker"]
        )

        # Native Polars dtype enforcement (no pandas conversion needed)
        float_cols = [
            "stock30_close_return",
            "stock30_rolling_vol_5",
            "opt30_strike",
            "opt30_moneyness",
            "opt30_mid_price_return",
            "opt30_bid_ask_spread",
            "opt30_implied_volatility",
            "opt30_delta",
            "opt30_theta",
            "opt30_volume_return",
            "opt30_rolling_vol_5",
            "opt30_call_flow",
            "opt30_put_flow",
            "opt30_flow_divergence",
            "opt30_net_gamma",
            "opt30_gamma_mean_5",
            "opt30_gamma_std_5",
        ]

        bool_cols = ["stock30_is_last_30min", "opt30_gamma_squeeze"]

        df = df.with_columns(
            [pl.col(col).cast(pl.Float64) for col in float_cols if col in df.columns]
            + [pl.col(col).cast(pl.Boolean) for col in bool_cols if col in df.columns]
        )

        # Log statistics
        print("\nProcessing Statistics:")
        print(f"Total rows: {df.shape[0]}")
        print(f"Total columns: {df.shape[1]}")

        print("\nJoin Coverage:")
        stocks_count = stocks_data.shape[0]
        options_count = options_data.shape[0]
        joined_count = df.shape[0]
        print(f"Stocks rows: {stocks_count}")
        print(f"Options rows: {options_count}")
        print(f"Joined rows: {joined_count}")
        print(f"Join percentage: {(joined_count / stocks_count) * 100:.2f}% of stocks")

        # Advanced signal coverage
        if "opt30_flow_divergence" in df.columns:
            flow_non_null = df.filter(
                pl.col("opt30_flow_divergence").is_not_null()
            ).shape[0]
            print(
                f"Flow divergence coverage: {flow_non_null}/{joined_count} ({(flow_non_null/joined_count)*100:.1f}%)"
            )

        if "opt30_gamma_squeeze" in df.columns:
            squeeze_non_null = df.filter(
                pl.col("opt30_gamma_squeeze").is_not_null()
            ).shape[0]
            squeeze_true = df.filter(pl.col("opt30_gamma_squeeze") == True).shape[0]
            print(
                f"Gamma squeeze coverage: {squeeze_non_null}/{joined_count} ({(squeeze_non_null/joined_count)*100:.1f}%)"
            )
            print(
                f"Gamma squeeze signals: {squeeze_true}/{joined_count} ({(squeeze_true/joined_count)*100:.1f}%)"
            )

        # Additional statistics
        print("\nUnique Counts:")
        print(f"Unique tickers in stocks: {stocks_data['ticker'].n_unique()}")
        print(f"Unique tickers in options: {options_data['ticker'].n_unique()}")
        print(f"Unique tickers in joined data: {df['ticker'].n_unique()}")

        # Time range statistics
        print("\nTimestamp Ranges:")
        stocks_times = stocks_data["timestamp"].unique().sort()
        options_times = options_data["timestamp"].unique().sort()
        print(f"Stocks: {stocks_times[0]} to {stocks_times[-1]}")
        print(f"Options: {options_times[0]} to {options_times[-1]}")

        print("\nNull counts:")
        for col in df.columns:
            null_count = df[col].null_count()
            null_pct = (null_count / df.shape[0]) * 100
            print(f"{col}: {null_count} nulls ({null_pct:.2f}%)")

        if not dry_run:
            # Write the joined data with optimized compression
            df.write_parquet(
                output_path,
                compression="zstd",
                statistics=True,
                use_pyarrow=True,
                pyarrow_options={"compression_level": 3},
            )
            print(f"\nWrote optimized joined data to: {output_path}")
        else:
            print("\nDRY RUN: Skipping file write")

        print(f"Successfully built intraday master dataset with {df.shape[0]} rows")

        return {
            "status": "success",
            "date": date,
            "rows_processed": df.shape[0],
            "join_coverage_pct": (joined_count / stocks_count) * 100,
            "output_path": output_path if not dry_run else None,
            "statistics": {
                "stocks": {
                    "total_rows": stocks_count,
                    "unique_tickers": stocks_data["ticker"].n_unique(),
                    "time_range": [str(stocks_times[0]), str(stocks_times[-1])],
                },
                "options": {
                    "total_rows": options_count,
                    "unique_tickers": options_data["ticker"].n_unique(),
                    "time_range": [str(options_times[0]), str(options_times[-1])],
                },
                "joined": {
                    "total_rows": joined_count,
                    "unique_tickers": df["ticker"].n_unique(),
                },
            },
        }

    except Exception as e:
        print(f"Error building intraday master dataset: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build intraday master dataset")
    parser.add_argument(
        "--date", type=str, required=True, help="Processing date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without writing files"
    )

    args = parser.parse_args()
    result = run(args.date, dry_run=args.dry_run)
    print(f"Task result: {result}")
