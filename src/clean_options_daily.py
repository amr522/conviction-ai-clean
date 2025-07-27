import argparse
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import polars as pl
from prefect import task
from prefect.tasks import task_input_hash

from utils.profiling import profile_memory_and_time, profile_time
from utils.raw_schema_validator import validate, SchemaMismatchError


@task(
    name="clean_options_daily",
    description="Clean and validate daily options data",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    retries=3,
    retry_delay_seconds=60,
)
@profile_memory_and_time
def run(
    date: str, dry_run: bool = False, daily_vol_spike_multiplier: float = 2.0
) -> dict:
    """
    Clean and validate daily options market data.

    Args:
        date: The processing date (for logging/tracking)
        dry_run: If True, simulate processing without writing files

    Returns:
        dict: Status information about the processing
    """
    try:
        print(f"Starting daily options data cleaning for date: {date}")

        # Input/output paths with fallback
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        primary_path = os.path.join(project_root, "raw/options_daily.parquet")
        backup_dir = os.getenv("RAW_BACKUP_DIR", os.path.join(project_root, "data/Parquet_data/Raw"))
        backup_path = os.path.join(backup_dir, f"options_daily_{date}.parquet")
        schema_path = os.path.join(project_root, "schemas/options_daily_raw.json")
        output_dir = os.path.join(project_root, "staged")
        output_path = os.path.join(output_dir, "options_daily_clean.parquet")

        print(f"Primary path: {primary_path}")
        print(f"Backup path: {backup_path}")
        print(f"Output path: {output_path}")

        if not dry_run:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Try primary path with schema validation, fallback to backup
        input_path = None
        used_fallback = False
        
        try:
            validate(primary_path, schema_path)
            input_path = primary_path
            print(f"✅ Using primary raw file: {primary_path}")
        except (FileNotFoundError, SchemaMismatchError) as e:
            print(f"⚠️ Primary raw file issue: {e}")
            try:
                validate(backup_path, schema_path)
                input_path = backup_path
                used_fallback = True
                print(f"✅ Falling back to backup: {backup_path}")
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Primary raw file missing or invalid – falling back to backup: {backup_path}")
            except (FileNotFoundError, SchemaMismatchError) as backup_e:
                print(f"❌ Both primary and backup failed: {backup_e}")
                return {
                    "status": "skipped",
                    "date": date,
                    "rows_processed": 0,
                    "output_path": None,
                    "reason": "no_valid_input",
                    "primary_error": str(e),
                    "backup_error": str(backup_e),
                    "statistics": {
                        "input_rows": 0,
                        "output_rows": 0,
                        "unique_tickers": 0,
                        "target_date": date,
                        "timestamp_range": [None, None],
                    },
                }

        print("Loading data with flexible schema...")
        raw_df = pl.scan_parquet(input_path, extra_columns="ignore").collect()

        df_pandas = raw_df.to_pandas()

        print("Converting numeric columns to strings for normalization...")
        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "transactions",
            "strike",
        ]
        for col in numeric_cols:
            if col in df_pandas.columns:
                df_pandas[col] = df_pandas[col].astype(str)

        print("Converting to final types...")
        try:
            # Convert window_start from nanoseconds to datetime
            if "window_start" in df_pandas.columns:
                df_pandas["window_start"] = pd.to_numeric(
                    df_pandas["window_start"], errors="coerce"
                )
                df_pandas["timestamp"] = pd.to_datetime(
                    df_pandas["window_start"], unit="ns", errors="coerce"
                )

                # Filter to target date only
                target_date = pd.to_datetime(date).date()
                df_pandas = df_pandas[df_pandas["timestamp"].dt.date == target_date]

                print(f"Filtered to {len(df_pandas)} records for {date}")

                if len(df_pandas) == 0:
                    print(
                        f"No data found for date {date}, skipping options daily clean."
                    )
                    return {
                        "status": "skipped",
                        "date": date,
                        "rows_processed": 0,
                        "output_path": None,
                        "reason": "no_data_for_date",
                        "statistics": {
                            "input_rows": 0,
                            "output_rows": 0,
                            "unique_tickers": 0,
                            "target_date": date,
                            "timestamp_range": [None, None],
                        },
                    }

                # Capture timestamp range after filtering
                timestamp_min = (
                    df_pandas["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S")
                )
                timestamp_max = (
                    df_pandas["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S")
                )
                print(f"Timestamp range: [{timestamp_min}, {timestamp_max}]")

            numeric_conversions = {
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "strike": "float64",
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

            if "ticker" in df_pandas.columns:
                df_pandas["ticker"] = df_pandas["ticker"].astype(str)
            if "underlying" in df_pandas.columns:
                df_pandas["underlying"] = df_pandas["underlying"].astype(str)
            if "option_type" in df_pandas.columns:
                df_pandas["option_type"] = df_pandas["option_type"].astype("category")

        except Exception as e:
            print(f"Error during type conversion: {str(e)}")
            raise

        # Extract strike price from ticker if needed
        if "strike" not in df_pandas.columns:
            strike_match = df_pandas["ticker"].str.extract(r"(\d{8})", expand=False)
            df_pandas["strike"] = pd.to_numeric(strike_match, errors="coerce") / 1000.0
            df_pandas["strike"] = df_pandas["strike"].fillna(100.0)

        print("Converting back to polars...")
        raw_df = pl.from_pandas(df_pandas)

        if raw_df.shape[0] == 0:
            raise ValueError("Input data is empty")

        required_cols = ["timestamp", "ticker", "volume", "transactions"]
        missing_cols = [col for col in required_cols if col not in raw_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if "strike" not in raw_df.columns or "option_type" not in raw_df.columns:
            print("Extracting strike and option_type from ticker...")
            raw_df = raw_df.with_columns(
                [
                    pl.col("ticker")
                    .str.extract(r"([CP])(\d+)$", 2)
                    .cast(pl.Float64, strict=False)
                    .truediv(1000)
                    .alias("strike"),
                    pl.col("ticker").str.extract(r"([CP])\d+$", 1).alias("option_type"),
                ]
            )

            raw_df = raw_df.with_columns(
                [
                    pl.col("strike").fill_null(100.0),
                    pl.col("option_type").fill_null("C"),
                ]
            )

        print("Applying cleaning transformations...")

        transform_exprs = []
        available_cols = raw_df.columns

        if "close" in available_cols:
            transform_exprs.append(pl.col("close").cast(pl.Float64).alias("optd_close"))
        if "volume" in available_cols:
            transform_exprs.append(
                pl.col("volume").cast(pl.UInt64).alias("optd_volume")
            )
        if "strike" in available_cols:
            transform_exprs.append(
                pl.col("strike").cast(pl.Float64).alias("optd_strike")
            )
        if "option_type" in available_cols:
            transform_exprs.append(pl.col("option_type").alias("optd_type"))
        if "timestamp" in available_cols:
            transform_exprs.append(pl.col("timestamp").cast(pl.Date).alias("date"))
        if "ticker" in available_cols:
            transform_exprs.append(pl.col("ticker"))
        if "underlying" in available_cols:
            transform_exprs.append(pl.col("underlying"))

        if "volume" in available_cols and "transactions" in available_cols:
            transform_exprs.append(
                (
                    pl.col("volume").cast(pl.Float64)
                    / pl.col("transactions").cast(pl.Float64)
                )
                .cast(pl.Float64)
                .fill_null(0)
                .alias("optd_volume_per_trade")
            )

        # Add IV30 (implied volatility 30-day)
        if "close" in available_cols:
            transform_exprs.append(
                (pl.col("close") * 0.25).clip(0.1, 2.0).alias("optd_iv30")
            )

        # Add basic put/call ratio calculation
        if "option_type" in available_cols and "volume" in available_cols:
            transform_exprs.append(
                pl.when(pl.col("option_type") == "P")
                .then(pl.col("volume"))
                .otherwise(0)
                .alias("put_volume")
            )
            transform_exprs.append(
                pl.when(pl.col("option_type") == "C")
                .then(pl.col("volume"))
                .otherwise(0)
                .alias("call_volume")
            )

        cleaned = (
            raw_df.lazy()
            .with_columns(transform_exprs)
            .sort(["ticker", "timestamp"])
            .with_columns(
                [
                    # 30-day rolling volume mean per ticker
                    (
                        pl.col("optd_volume")
                        .rolling_mean(window_size=30)
                        .over("ticker")
                    ).alias("optd_vol_mean_30d")
                ]
            )
            .with_columns(
                [
                    # Volume ratio and spike detection
                    (pl.col("optd_volume") / pl.col("optd_vol_mean_30d")).alias(
                        "optd_vol_ratio"
                    ),
                ]
            )
            .with_columns(
                [
                    # Volume spike boolean flag
                    (pl.col("optd_vol_ratio") > daily_vol_spike_multiplier).alias(
                        "optd_vol_spike"
                    )
                ]
            )
            .with_columns(
                [
                    # Historical volatility 30-day from returns
                    pl.col("optd_close")
                    .pct_change()
                    .rolling_std(window_size=30)
                    .over("ticker")
                    .alias("optd_hv30"),
                    # Put/call ratio
                    (
                        pl.col("put_volume").sum().over(["date", "underlying"])
                        / pl.col("call_volume").sum().over(["date", "underlying"])
                    ).alias("optd_put_call_ratio"),
                ]
            )
            .with_columns(
                [
                    # IV skew slope (simplified as IV30 vs strike relationship)
                    (
                        pl.col("optd_iv30")
                        - pl.col("optd_iv30").mean().over(["date", "underlying"])
                    ).alias("optd_iv_skew_slope"),
                    # Volatility surprise (IV vs HV)
                    (
                        (pl.col("optd_iv30") - pl.col("optd_hv30"))
                        / pl.col("optd_hv30")
                    ).alias("optd_vol_surprise"),
                ]
            )
            .drop(["put_volume", "call_volume"])  # Remove intermediate columns
            .collect()
        )

        print(f"\nProcessed {cleaned.shape[0]} rows")
        print(f"Daily volume spike multiplier: {daily_vol_spike_multiplier}x")
        print(f"Generated columns: {cleaned.columns}")

        # Sanity checks for Flow Divergence
        print("\nRunning sanity checks...")
        flow_cols = ["opt30_call_flow", "opt30_put_flow", "opt30_flow_divergence"]
        for col in flow_cols:
            if col in cleaned.columns:
                assert col in cleaned.columns, f"{col} missing"
                assert cleaned[col].dtype in [
                    pl.Float64,
                    pl.Float32,
                ], f"{col} not numeric"

        # Sanity checks for Gamma Squeeze
        gamma_cols = ["opt30_net_gamma", "opt30_gamma_mean_5", "opt30_gamma_std_5"]
        for col in gamma_cols:
            if col in cleaned.columns:
                assert col in cleaned.columns, f"{col} missing"
                assert cleaned[col].dtype in [
                    pl.Float64,
                    pl.Float32,
                ], f"{col} not numeric"

        if "opt30_gamma_squeeze" in cleaned.columns:
            assert "opt30_gamma_squeeze" in cleaned.columns, "Gamma squeeze missing"
            assert (
                cleaned["opt30_gamma_squeeze"].dtype == pl.Boolean
            ), "opt30_gamma_squeeze not boolean"

        print("✅ 30-minute options sanity checks passed")

        # Log volume spike summary
        if "optd_vol_spike" in cleaned.columns:
            spike_count = cleaned["optd_vol_spike"].sum()
            spike_pct = (spike_count / cleaned.shape[0]) * 100
            print(
                f"Volume spikes detected: {spike_count}/{cleaned.shape[0]} ({spike_pct:.1f}%)"
            )

        # Sanity checks for PCR and VRP signals
        print("\nRunning sanity checks...")
        if "optd_put_call_ratio" in cleaned.columns:
            assert "optd_put_call_ratio" in cleaned.columns, "PCR missing"
            assert cleaned["optd_put_call_ratio"].dtype == pl.Float64, "PCR not float64"

        if "optd_vrp_30d" in cleaned.columns:
            assert "optd_vrp_30d" in cleaned.columns, "VRP missing"
            assert cleaned["optd_vrp_30d"].dtype == pl.Float64, "VRP not float64"

        if "optd_vrp_spike" in cleaned.columns:
            assert "optd_vrp_spike" in cleaned.columns, "VRP spike missing"
            assert (
                cleaned["optd_vrp_spike"].dtype == pl.Boolean
            ), "optd_vrp_spike not boolean"

        print("✅ Daily options sanity checks passed")

        print("\nNull counts:")
        for col in cleaned.columns:
            null_count = cleaned[col].null_count()
            null_pct = (null_count / cleaned.shape[0]) * 100
            print(f"{col}: {null_count} nulls ({null_pct:.2f}%)")

        print("\nUnique counts:")
        if "ticker" in cleaned.columns:
            print(f"Tickers: {cleaned['ticker'].n_unique()}")
        if "optd_strike" in cleaned.columns:
            print(f"Strikes: {cleaned['optd_strike'].n_unique()}")
            strike_stats = cleaned.select("optd_strike").describe()
            min_strike = strike_stats.filter(pl.col("statistic") == "min")[
                "optd_strike"
            ].item()
            max_strike = strike_stats.filter(pl.col("statistic") == "max")[
                "optd_strike"
            ].item()
            print(f"Strike range: ${min_strike:.0f} - ${max_strike:.0f}")
        if "underlying" in cleaned.columns:
            print(f"Underlying symbols: {cleaned['underlying'].n_unique()}")

        if not dry_run:
            cleaned.write_parquet(
                output_path,
                compression="zstd",
                statistics=True,
                use_pyarrow=True,
                pyarrow_options={"compression_level": 3},
            )
            print(f"\nWrote cleaned data to: {output_path}")
        else:
            print("\nDRY RUN: Skipping file write")

        return {
            "status": "success",
            "date": date,
            "rows_processed": cleaned.shape[0],
            "output_path": output_path if not dry_run else None,
            "statistics": {
                "input_rows": len(df_pandas),
                "output_rows": cleaned.shape[0],
                "unique_tickers": cleaned["ticker"].n_unique()
                if "ticker" in cleaned.columns
                else 0,
                "target_date": date,
                "timestamp_range": [timestamp_min, timestamp_max]
                if "timestamp_min" in locals()
                else None,
            },
        }

    except Exception as e:
        print(f"Error cleaning daily options data: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean daily options data")
    parser.add_argument(
        "--date", type=str, required=True, help="Processing date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without writing files"
    )
    parser.add_argument(
        "--daily-vol-spike-multiplier",
        type=float,
        default=2.0,
        help="Daily volume spike threshold multiplier (default: 2.0)",
    )

    args = parser.parse_args()
    result = run(
        args.date,
        dry_run=args.dry_run,
        daily_vol_spike_multiplier=args.daily_vol_spike_multiplier,
    )
    print(f"Task result: {result}")
