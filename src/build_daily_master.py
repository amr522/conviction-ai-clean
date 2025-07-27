import argparse
import os
from datetime import timedelta
from pathlib import Path

import polars as pl
from prefect import task
from prefect.tasks import task_input_hash


@task(
    name="build_daily_master",
    description="Build daily master dataset by joining stocks and options data",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    retries=3,
    retry_delay_seconds=60,
)
def run(
    date: str,
    dry_run: bool = False,
    use_raw_macro: bool = False,
    raw_fred_csv: str = None,
    raw_vix_json: str = None,
    raw_dxy_csv: str = None,
    raw_news_dir: str = None,
) -> dict:
    """
    Build daily master dataset by joining stocks and options data.

    Args:
        date: The processing date (for logging/tracking)
        dry_run: If True, simulate processing without writing files

    Returns:
        dict: Status information about the processing
    """
    try:
        print(f"Starting daily master dataset build for date: {date}")

        # Input/output paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stocks_path = os.path.join(project_root, "staged/stocks_daily_clean.parquet")
        options_path = os.path.join(project_root, "staged/options_daily_clean.parquet")
        output_dir = os.path.join(project_root, "staged")
        output_path = os.path.join(output_dir, "daily_master.parquet")

        # Macro data paths
        fred_parquet_path = os.path.join(project_root, "data/Parquet_data/fred.parquet")
        vix_parquet_path = os.path.join(
            project_root, "data/Parquet_data/vix_data.parquet"
        )
        dxy_parquet_path = os.path.join(project_root, "data/Parquet_data/dxy.parquet")
        news_parquet_path = os.path.join(
            project_root, "data/Parquet_data/news_data.parquet"
        )

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

        # Define expected schemas for input validation
        stocks_schema = {
            "date": pl.Date,
            "ticker": pl.Utf8,
            "stockd_close": pl.Float64,
            "stockd_volume": pl.UInt64,
            "stockd_return_1d": pl.Float64,
            "stockd_return_1d_lag1": pl.Float64,
            "stockd_vol_7d": pl.Float64,
            "stockd_vol_7d_lag1": pl.Float64,
        }

        options_schema = {
            "date": pl.Date,
            "ticker": pl.Utf8,
            "optd_strike": pl.Float64,
            "optd_type": pl.Categorical,
            "optd_volume": pl.UInt64,
            "optd_moneyness": pl.Float64,
        }

        # Load and validate stocks data with schema
        print("\nLoading stocks data with schema validation...")
        try:
            # Create schema cast expressions
            stock_casts = [
                pl.col(col).cast(dtype, strict=False).alias(col)
                for col, dtype in stocks_schema.items()
            ]

            # Load and validate in one pass using lazy evaluation
            stocks_data = (
                pl.scan_parquet(stocks_path, extra_columns="ignore")
                .with_columns(stock_casts)
                .collect(streaming=True)
            )

            if stocks_data.shape[0] == 0:
                raise ValueError("Stocks data is empty")
            print(f"Stocks shape: {stocks_data.shape}")

            # Validate required columns
            required_stock_cols = ["date", "ticker", "stockd_close", "stockd_volume"]
            missing_cols = [
                col for col in required_stock_cols if col not in stocks_data.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns in stocks data: {missing_cols}"
                )

            # Check for null values in critical columns
            null_cols = [
                col for col in required_stock_cols if stocks_data[col].null_count() > 0
            ]
            if null_cols:
                print("Warning: Found null values in critical stock columns:")
                for col in null_cols:
                    print(f"  {col}: {stocks_data[col].null_count()} nulls")

        except Exception as e:
            raise ValueError(f"Error processing stocks data: {str(e)}")

        # Load and validate options data with schema
        print("\nLoading options data with schema validation...")
        try:
            # Create schema cast expressions
            option_casts = [
                pl.col(col).cast(dtype, strict=False).alias(col)
                for col, dtype in options_schema.items()
            ]

            # Load and validate in one pass using lazy evaluation
            options_data = (
                pl.scan_parquet(options_path, extra_columns="ignore")
                .with_columns(option_casts)
                .collect(streaming=True)
            )

            if options_data.shape[0] == 0:
                raise ValueError("Options data is empty")
            print(f"Options shape: {options_data.shape}")

            # Validate required columns
            required_option_cols = ["date", "ticker", "optd_strike", "optd_type"]
            missing_cols = [
                col for col in required_option_cols if col not in options_data.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns in options data: {missing_cols}"
                )

            # Check for null values in critical columns
            null_cols = [
                col
                for col in required_option_cols
                if options_data[col].null_count() > 0
            ]
            if null_cols:
                print("Warning: Found null values in critical options columns:")
                for col in null_cols:
                    print(f"  {col}: {options_data[col].null_count()} nulls")

        except Exception as e:
            raise ValueError(f"Error processing options data: {str(e)}")

        # Load macro data
        print("\nLoading macro data...")
        from clean_macro_data import load_data_source

        fred_df = load_data_source(
            "FRED", raw_fred_csv, fred_parquet_path, use_raw_macro
        )
        vix_df = load_data_source(
            "VIX", raw_vix_json, vix_parquet_path, use_raw_macro, is_json=True
        )
        dxy_df = load_data_source("DXY", raw_dxy_csv, dxy_parquet_path, use_raw_macro)
        news_df = load_data_source(
            "News", raw_news_dir, news_parquet_path, use_raw_macro
        )

        # Convert to Polars and rename columns with prefixes
        macro_dfs = {}

        if not fred_df.empty:
            fred_pl = pl.from_pandas(fred_df)
            fred_pl = fred_pl.rename(
                {col: f"fred_{col}" for col in fred_pl.columns if col != "date"}
            )
            macro_dfs["fred"] = fred_pl
            print(f"FRED data: {fred_pl.shape}")

        if not vix_df.empty:
            vix_pl = pl.from_pandas(vix_df)
            vix_pl = vix_pl.rename(
                {col: f"vix_{col}" for col in vix_pl.columns if col != "date"}
            )
            macro_dfs["vix"] = vix_pl
            print(f"VIX data: {vix_pl.shape}")

        if not dxy_df.empty:
            dxy_pl = pl.from_pandas(dxy_df)
            dxy_pl = dxy_pl.rename(
                {col: f"dxy_{col}" for col in dxy_pl.columns if col != "date"}
            )
            macro_dfs["dxy"] = dxy_pl
            print(f"DXY data: {dxy_pl.shape}")

        if not news_df.empty:
            news_pl = pl.from_pandas(news_df)
            # Aggregate news by date
            news_agg = news_pl.group_by("date").agg(
                [
                    pl.len().alias("news_count"),
                    pl.col("sentiment").mean().alias("news_avg_sentiment")
                    if "sentiment" in news_pl.columns
                    else pl.lit(0.0).alias("news_avg_sentiment"),
                ]
            )
            macro_dfs["news"] = news_agg
            print(f"News data: {news_agg.shape}")

        # Define expected dtypes for the joined dataset
        master_dtypes = {
            # Time and identifier columns
            "date": pl.Date,
            "ticker": pl.Utf8,
            # Stock features
            "stockd_close": pl.Float64,
            "stockd_volume": pl.UInt64,
            "stockd_return_1d": pl.Float64,
            "stockd_return_1d_lag1": pl.Float64,
            "stockd_vol_7d": pl.Float64,
            "stockd_vol_7d_lag1": pl.Float64,
            "stockd_volume_pct_change": pl.Float64,
            "stockd_beta_spy": pl.Float64,
            "stockd_days_to_earnings": pl.Int64,
            "stockd_earnings_flag": pl.Boolean,
            # Options features
            "optd_strike": pl.Float64,
            "optd_moneyness": pl.Float64,
            "optd_iv30": pl.Float64,
            "optd_hv30": pl.Float64,
            "optd_iv30_lag1": pl.Float64,
            "optd_hv30_lag1": pl.Float64,
            "optd_iv_percentile": pl.Float64,
            "optd_iv_percentile_lag1": pl.Float64,
            "optd_vrp_30d": pl.Float64,
            "optd_vrp_30d_lag1": pl.Float64,
            "optd_iv_skew_slope": pl.Float64,
            "optd_vol_surprise": pl.Float64,
            "optd_put_call_ratio": pl.Float64,
            "optd_chain_liquidity_lag1": pl.Float64,
            "optd_volume": pl.UInt64,
            "optd_smile_convexity": pl.Float64,
            "optd_ivrv_spread": pl.Float64,
            # Macro features
            "fred_fed_funds_rate": pl.Float64,
            "fred_unemployment_rate": pl.Float64,
            "fred_inflation_rate": pl.Float64,
            "vix_index": pl.Float64,
            "vix_close": pl.Float64,
            "dxy_rate": pl.Float64,
            "dxy_close": pl.Float64,
            "news_count": pl.UInt64,
            "news_avg_sentiment": pl.Float64,
        }

        # Join datasets with optimized streaming
        print("\nJoining datasets with type validation...")
        try:
            # Pre-select only needed columns to optimize memory
            stock_cols = ["date", "ticker"] + [
                col for col in stocks_data.columns if col.startswith("stockd_")
            ]
            option_cols = ["date", "ticker"] + [
                col for col in options_data.columns if col.startswith("optd_")
            ]

            # Create streaming lazy frames
            stocks_lazy = stocks_data.select(stock_cols).lazy()

            options_lazy = options_data.select(option_cols).lazy()

            # Prepare type casting expressions
            cast_expressions = []
            for col, dtype in master_dtypes.items():
                if col.startswith("stockd_") or col.startswith("optd_"):
                    cast_expressions.append(
                        pl.col(col).cast(dtype, strict=False).alias(col)
                    )

            # Perform streaming join with macro data
            df = stocks_lazy.join(
                options_lazy, on=["date", "ticker"], how="inner"
            ).collect(streaming=True)

            # Join macro data
            for name, macro_df in macro_dfs.items():
                print(f"Joining {name} data...")
                df = df.join(macro_df, on="date", how="left")
                print(f"After {name} join: {df.shape}")

            # Convert to pandas for comprehensive dtype enforcement
            import pandas as pd

            df_final = df.to_pandas()

            # Enforce feature dtypes to match finalized schema
            feature_dtypes = {
                # stocks daily
                "stockd_close": "float64",
                "stockd_volume": "UInt64",
                "stockd_return_1d": "float64",
                "stockd_return_1d_lag1": "float64",
                "stockd_vol_7d": "float64",
                "stockd_vol_7d_lag1": "float64",
                "stockd_volume_pct_change": "float64",
                "stockd_beta_spy": "float64",
                "stockd_days_to_earnings": "Int64",
                # options daily
                "optd_strike": "float64",
                "optd_moneyness": "float64",
                "optd_iv30": "float64",
                "optd_hv30": "float64",
                "optd_iv30_lag1": "float64",
                "optd_hv30_lag1": "float64",
                "optd_iv_percentile": "float64",
                "optd_iv_percentile_lag1": "float64",
                "optd_vrp_30d": "float64",
                "optd_vrp_30d_lag1": "float64",
                "optd_iv_skew_slope": "float64",
                "optd_vol_surprise": "float64",
                "optd_smile_convexity": "float64",
                "optd_ivrv_spread": "float64",
                "optd_put_call_ratio": "float64",
                "optd_chain_liquidity_lag1": "float64",
                "optd_volume": "UInt64",
                # options 30min
                "opt30_strike": "float64",
                "opt30_moneyness": "float64",
                "opt30_mid_price_return": "float64",
                "opt30_bid_ask_spread": "float64",
                "opt30_implied_volatility": "float64",
                "opt30_delta": "float64",
                "opt30_theta": "float64",
                "opt30_volume_return": "float64",
                "opt30_open_interest_change": "float64",
                "opt30_rolling_vol_5": "float64",
                # macro features
                "fred_fed_funds_rate": "float64",
                "fred_unemployment_rate": "float64",
                "fred_inflation_rate": "float64",
                "vix_index": "float64",
                "vix_close": "float64",
                "dxy_rate": "float64",
                "dxy_close": "float64",
                "news_count": "UInt64",
                "news_avg_sentiment": "float64",
            }

            for col, dtype in feature_dtypes.items():
                if col in df_final.columns:
                    df_final[col] = df_final[col].astype(dtype)

            # Convert back to polars
            df = pl.from_pandas(df_final)

            # Quick validation of join results
            if df.shape[0] == 0:
                raise ValueError("Join resulted in empty dataset")

            # Check for unexpected null values
            critical_cols = (
                ["date", "ticker"] + required_stock_cols + required_option_cols
            )
            null_cols = [
                col
                for col in critical_cols
                if col in df.columns and df[col].null_count() > 0
            ]

            if null_cols:
                print("\nWarning: Found null values in critical columns after join:")
                for col in null_cols:
                    print(f"  {col}: {df[col].null_count()} nulls")

        except Exception as e:
            raise ValueError(f"Error during dataset join: {str(e)}")

        # Log statistics
        print("\nProcessing Statistics:")
        print(f"Total rows: {df.shape[0]}")
        print(f"Total columns: {df.shape[1]}")

        print("\nData Type Validation:")
        for col in df.columns:
            if col in master_dtypes:
                actual_type = df[col].dtype
                expected_type = master_dtypes[col]
                if actual_type != expected_type:
                    print(
                        f"WARNING: Column {col} has type {actual_type}, expected {expected_type}"
                    )

        print("\nJoin Coverage:")
        stocks_count = stocks_data.shape[0]
        options_count = options_data.shape[0]
        joined_count = df.shape[0]
        print(f"Stocks rows: {stocks_count}")
        print(f"Options rows: {options_count}")
        print(f"Joined rows: {joined_count}")
        print(f"Join percentage: {(joined_count / stocks_count) * 100:.2f}% of stocks")

        # Additional statistics
        print("\nUnique Counts:")
        print(f"Unique tickers in stocks: {stocks_data['ticker'].n_unique()}")
        print(f"Unique tickers in options: {options_data['ticker'].n_unique()}")
        print(f"Unique tickers in joined data: {df['ticker'].n_unique()}")

        # Date range statistics
        print("\nDate Ranges:")
        stocks_dates = stocks_data["date"].unique().sort()
        options_dates = options_data["date"].unique().sort()
        print(f"Stocks: {stocks_dates[0]} to {stocks_dates[-1]}")
        print(f"Options: {options_dates[0]} to {options_dates[-1]}")

        print("\nNull counts:")
        for col in df.columns:
            null_count = df[col].null_count()
            null_pct = (null_count / df.shape[0]) * 100
            print(f"{col}: {null_count} nulls ({null_pct:.2f}%)")

        if not dry_run:
            try:
                print("\nPreparing output with final schema validation...")

                # Create optimized final schema for writing
                final_schema = {
                    col: dtype
                    for col, dtype in master_dtypes.items()
                    if col in df.columns
                }

                # Create type casting expressions with null handling
                final_cast_expressions = []
                for col, dtype in final_schema.items():
                    expr = pl.col(col).cast(dtype, strict=False)
                    if dtype in [pl.Float64, pl.Int64, pl.UInt64]:
                        expr = expr.fill_null(0)
                    elif dtype == pl.Boolean:
                        expr = expr.fill_null(False)
                    elif dtype == pl.Utf8:
                        expr = expr.fill_null("")
                    final_cast_expressions.append(expr.alias(col))

                # Apply final type casting
                print("Applying final type conversions...")
                df = df.with_columns(final_cast_expressions)

                # Verify all columns have correct types
                type_mismatches = []
                for col, dtype in final_schema.items():
                    if df[col].dtype != dtype:
                        type_mismatches.append(
                            f"{col}: got {df[col].dtype}, expected {dtype}"
                        )

                if type_mismatches:
                    raise ValueError(
                        f"Type validation failed:\n" + "\n".join(type_mismatches)
                    )

                # Write with optimized settings and chunking
                print(f"Writing data to: {output_path}")
                df.write_parquet(
                    output_path,
                    compression="zstd",
                    compression_level=3,
                    statistics=True,
                    use_pyarrow=True,
                    row_group_size=100000,  # Optimize for ~100MB chunks
                )

                # Quick validation of written file
                print("Verifying written file...")
                verify_df = pl.scan_parquet(output_path).collect()

                if verify_df.shape != df.shape:
                    raise ValueError(
                        f"Written file validation failed: "
                        f"shape mismatch {verify_df.shape} vs {df.shape}"
                    )

                print("Successfully wrote and verified output file")

            except Exception as e:
                raise ValueError(f"Error in output processing: {str(e)}")
        else:
            print("\nDRY RUN: Skipping file write")

        print(f"Successfully built daily master dataset with {df.shape[0]} rows")

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
                    "date_range": [str(stocks_dates[0]), str(stocks_dates[-1])],
                },
                "options": {
                    "total_rows": options_count,
                    "unique_tickers": options_data["ticker"].n_unique(),
                    "date_range": [str(options_dates[0]), str(options_dates[-1])],
                },
                "joined": {
                    "total_rows": joined_count,
                    "unique_tickers": df["ticker"].n_unique(),
                },
            },
        }

    except Exception as e:
        if "data type mismatch" in str(e).lower():
            print(f"Schema error - data type mismatch detected: {str(e)}")
            print(
                "This usually means one of the cleaning scripts is not enforcing proper data types."
            )
            print(
                "Please verify that all cleaning scripts are using correct type casting."
            )
        else:
            print(f"Error building daily master dataset: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build daily master dataset")
    parser.add_argument(
        "--date", type=str, required=True, help="Processing date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without writing files"
    )
    parser.add_argument(
        "--use-raw-macro",
        action="store_true",
        help="Force use of raw macro data sources",
    )
    parser.add_argument(
        "--raw-fred-csv",
        default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/FRED.csv",
    )
    parser.add_argument(
        "--raw-vix-json",
        default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/vix_data.json",
    )
    parser.add_argument(
        "--raw-dxy-csv",
        default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/DXY.csv",
    )
    parser.add_argument(
        "--raw-news-dir",
        default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/news",
    )

    args = parser.parse_args()
    result = run(
        args.date,
        dry_run=args.dry_run,
        use_raw_macro=args.use_raw_macro,
        raw_fred_csv=args.raw_fred_csv,
        raw_vix_json=args.raw_vix_json,
        raw_dxy_csv=args.raw_dxy_csv,
        raw_news_dir=args.raw_news_dir,
    )
    print(f"Task result: {result}")
