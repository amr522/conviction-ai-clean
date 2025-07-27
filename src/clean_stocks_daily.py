import argparse
import glob
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import polars as pl
from prefect import task
from prefect.tasks import task_input_hash


@task(
    name="clean_stocks_daily",
    description="Clean and validate daily stock data",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    retries=3,
    retry_delay_seconds=60,
)
def run(date: str, dry_run: bool = False) -> dict:
    """
    Clean and transform daily stock data using robust type handling.

    Args:
        date: The processing date (for logging/tracking)
        dry_run: If True, simulate processing without writing files

    Returns:
        dict: Status information about the processing
    """
    try:
        print(f"Starting daily stocks cleaning for date: {date}")

        # Input/output paths
        input_pattern = "/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Stocks_daily/*.parquet"
        output_dir = "/Users/amroheidak/Desktop/conviction-ai-clean/staged"
        output_path = os.path.join(output_dir, "stocks_daily_clean.parquet")

        print(f"Input pattern: {input_pattern}")
        print(f"Output path: {output_path}")

        if not dry_run:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Get all parquet files
        parquet_files = glob.glob(input_pattern)
        print(f"Found {len(parquet_files)} parquet files")

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found matching {input_pattern}")

        # Read each file individually and normalize schemas
        dfs = []
        reference_schema = None

        for i, file_path in enumerate(parquet_files):
            print(
                f"Processing {os.path.basename(file_path)} ({i+1}/{len(parquet_files)})..."
            )

            df_part = pl.read_parquet(file_path)

            # Set reference schema from first file
            if reference_schema is None:
                reference_schema = df_part.columns
                print(f"Reference schema: {reference_schema}")

            # Ensure all files have the same columns in the same order
            if set(df_part.columns) != set(reference_schema):
                print(f"  Schema mismatch in {os.path.basename(file_path)}")
                print(f"  Expected: {reference_schema}")
                print(f"  Found: {df_part.columns}")

                # Add missing columns with null values
                for col in reference_schema:
                    if col not in df_part.columns:
                        df_part = df_part.with_columns(pl.lit(None).alias(col))

                # Remove extra columns
                df_part = df_part.select(reference_schema)

            # Reorder columns to match reference
            df_part = df_part.select(reference_schema)

            # Convert to pandas and normalize to strings
            df_pandas = df_part.to_pandas()
            for col in df_pandas.columns:
                df_pandas[col] = df_pandas[col].astype(str)

            df_normalized = pl.from_pandas(df_pandas)
            dfs.append(df_normalized)

        # Concatenate all normalized dataframes
        print("Concatenating all files...")
        raw_df = pl.concat(dfs, how="vertical_relaxed")

        print(f"Combined data shape: {raw_df.shape}")
        print(f"Available columns: {raw_df.columns}")

        # Convert to pandas for robust type handling
        df_pandas = raw_df.to_pandas()

        print("Converting to final types...")
        try:
            # Convert numeric columns
            numeric_conversions = {
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
            }

            for col, dtype in numeric_conversions.items():
                if col in df_pandas.columns:
                    df_pandas[col] = pd.to_numeric(
                        df_pandas[col], errors="coerce"
                    ).astype(dtype)

            if "volume" in df_pandas.columns:
                df_pandas["volume"] = (
                    pd.to_numeric(df_pandas["volume"], errors="coerce")
                    .fillna(0)
                    .astype("UInt64")
                )
            if "transactions" in df_pandas.columns:
                df_pandas["transactions"] = (
                    pd.to_numeric(df_pandas["transactions"], errors="coerce")
                    .fillna(0)
                    .astype("UInt32")
                )

            if "window_start" in df_pandas.columns:
                df_pandas["window_start"] = pd.to_numeric(
                    df_pandas["window_start"], errors="coerce"
                )
                df_pandas["window_start"] = pd.to_datetime(
                    df_pandas["window_start"], unit="ns", errors="coerce"
                )

            if "ticker" in df_pandas.columns:
                df_pandas["ticker"] = df_pandas["ticker"].astype(str)

        except Exception as e:
            print(f"Error during type conversion: {str(e)}")
            raise

        # Parse and filter by date
        total_raw = len(df_pandas)
        df_pandas["date"] = df_pandas["window_start"].dt.date
        target_date = pd.to_datetime(date).date()
        df_pandas = df_pandas.loc[df_pandas["date"] == target_date]
        filtered_count = len(df_pandas)

        print(
            f"Filtered daily stock data: {filtered_count}/{total_raw} rows for {date}"
        )

        if filtered_count == 0:
            raise ValueError(
                f"No data found for date {date}. Check available dates in the dataset."
            )

        # Capture timestamp range after filtering
        timestamp_min = df_pandas["window_start"].min().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_max = df_pandas["window_start"].max().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Timestamp range: [{timestamp_min}, {timestamp_max}]")

        # Convert back to polars
        print("Converting back to polars...")
        raw_df = pl.from_pandas(df_pandas)

        required_cols = ["window_start", "ticker", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in raw_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        print("Calculating features...")
        df = (
            raw_df.lazy()
            .with_columns(
                [
                    # Base features
                    pl.col("close").alias("stockd_close"),
                    pl.col("volume").alias("stockd_volume"),
                    pl.col("ticker").alias("ticker"),
                    pl.col("window_start").cast(pl.Date).alias("date"),
                ]
            )
            .with_columns(
                [
                    # Returns
                    (pl.col("stockd_close").shift() / pl.col("stockd_close") - 1)
                    .over("ticker")
                    .alias("stockd_return_1d")
                ]
            )
            .with_columns(
                [
                    # Volatility
                    pl.col("stockd_return_1d")
                    .rolling_std(window_size=7, min_periods=3)
                    .over("ticker")
                    .alias("stockd_vol_7d")
                ]
            )
            .with_columns(
                [
                    # Lagged features
                    pl.col("stockd_return_1d")
                    .shift()
                    .over("ticker")
                    .alias("stockd_return_1d_lag1"),
                    pl.col("stockd_vol_7d")
                    .shift()
                    .over("ticker")
                    .alias("stockd_vol_7d_lag1"),
                ]
            )
            .select(
                [
                    "date",
                    "ticker",
                    "stockd_close",
                    "stockd_volume",
                    "stockd_return_1d",
                    "stockd_vol_7d",
                    "stockd_return_1d_lag1",
                    "stockd_vol_7d_lag1",
                ]
            )
            .collect(streaming=True)
        )

        print(f"\nProcessed {df.shape[0]} rows")
        print(f"Generated columns: {df.columns}")

        print("\nNull counts:")
        for col in df.columns:
            null_count = df[col].null_count()
            null_pct = (null_count / df.shape[0]) * 100
            print(f"{col}: {null_count} nulls ({null_pct:.2f}%)")

        if not dry_run:
            df.write_parquet(
                output_path,
                compression="zstd",
                statistics=True,
                use_pyarrow=True,
                pyarrow_options={"compression_level": 3},
            )
            print(f"\nWrote cleaned data to: {output_path}")
        else:
            print("\nDRY RUN: Skipping file write")

        print(f"\nSuccessfully processed {df.shape[0]} rows of daily stock data")

        return {
            "status": "success",
            "date": date,
            "rows_processed": df.shape[0],
            "output_path": output_path if not dry_run else None,
            "statistics": {
                "input_rows": filtered_count,
                "output_rows": df.shape[0],
                "unique_tickers": df["ticker"].n_unique(),
                "target_date": date,
                "timestamp_range": [timestamp_min, timestamp_max],
            },
        }

    except Exception as e:
        print(f"Error processing daily stock data: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean daily stocks data")
    parser.add_argument(
        "--date", type=str, required=True, help="Processing date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without writing files"
    )

    args = parser.parse_args()
    result = run(args.date, dry_run=args.dry_run)
    print(f"Task result: {result}")
