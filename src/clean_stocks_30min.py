import polars as pl
import pandas as pd
from prefect import task
from prefect.tasks import task_input_hash
from datetime import timedelta
import os
from pathlib import Path
import argparse
from utils.profiling import profile_memory_and_time, profile_time

@task(
    name="clean_stocks_30min",
    description="Clean and validate 30-minute stock data",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    retries=3,
    retry_delay_seconds=60,
)
@profile_memory_and_time
def run(date: str, dry_run: bool = False) -> dict:
    """
    Clean and transform 30-minute stock market data.
    
    Args:
        date: The processing date (for logging/tracking)
        dry_run: If True, simulate processing without writing files
    
    Returns:
        dict: Status information about the processing
    """
    try:
        print(f"Starting 30-minute stocks data cleaning for date: {date}")
        
        # Input/output paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_path = os.path.join(project_root, "data/Parquet_data/stocks_minute")
        output_dir = os.path.join(project_root, "staged")
        output_path = os.path.join(output_dir, "stocks_30min_clean.parquet")
        
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")

        if not dry_run:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Validate input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # First read everything as strings
        read_schema = {
            "window_start": pl.Utf8,
            "ticker": pl.Utf8,
            "open": pl.Utf8,
            "high": pl.Utf8,
            "low": pl.Utf8,
            "close": pl.Utf8,
            "volume": pl.Utf8,
            "transactions": pl.Utf8
        }
        
        # Target schema after conversion
        schema = {
            "timestamp": pl.Datetime,
            "ticker": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.UInt64,
            "transactions": pl.UInt32
        }
        
        # Define feature schema
        feature_schema = {
            "stock30_close": pl.Float64,
            "stock30_volume": pl.UInt64,
            "stock30_close_return": pl.Float64,
            "stock30_rolling_vol_5": pl.Float64,
            "stock30_is_last_30min": pl.Boolean,
            "stock30_open": pl.Float64,
            "stock30_high": pl.Float64,
            "stock30_low": pl.Float64
        }
        
        # Load raw data with flexible schema
        print("Loading data with flexible schema...")
        raw_df = pl.scan_parquet(input_path, schema=None, extra_columns='ignore').collect()
        
        # Convert to pandas for reliable type coercion
        df_pandas = raw_df.to_pandas()
        
        print("Converting numeric columns to strings for normalization...")
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'transactions']
        for col in numeric_cols:
            if col in df_pandas.columns:
                df_pandas[col] = df_pandas[col].astype(str)
        
        print("Converting to final types...")
        try:
            # Convert window_start from nanoseconds to datetime
            df_pandas['window_start'] = pd.to_numeric(df_pandas['window_start'], errors='coerce')
            df_pandas['timestamp'] = pd.to_datetime(df_pandas['window_start'], unit='ns')
            
            # 1️⃣ Filter to target date only
            target_date = pd.to_datetime(date).date()
            df_pandas = df_pandas[df_pandas['timestamp'].dt.date == target_date]
            
            # 2️⃣ Restrict to regular trading hours (09:30–16:00)
            market_open = pd.to_datetime(f"{date} 09:30:00")
            market_close = pd.to_datetime(f"{date} 16:00:00")
            df_pandas = df_pandas[(df_pandas['timestamp'] >= market_open) & (df_pandas['timestamp'] < market_close)]
            
            print(f"Filtered to {len(df_pandas)} records for {date} trading hours")
            
            if len(df_pandas) == 0:
                raise ValueError(f"No data found for date {date}. Check available dates in the dataset.")
            
            # Capture timestamp range after filtering
            timestamp_min = df_pandas['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
            timestamp_max = df_pandas['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Timestamp range: [{timestamp_min}, {timestamp_max}]")
            
            # First convert numeric columns to their initial types
            df_pandas = df_pandas.astype({
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'Int64',  # Use nullable integer type first
                'transactions': 'Int64',  # Use nullable integer type first
            })
            # Convert to unsigned integers after initial conversion
            df_pandas['volume'] = df_pandas['volume'].fillna(0).astype('UInt64')
            df_pandas['transactions'] = df_pandas['transactions'].fillna(0).astype('UInt32')
            if 'ticker' in df_pandas.columns:
                df_pandas['ticker'] = df_pandas['ticker'].astype(str)
        except Exception as e:
            print(f"Error during type conversion: {str(e)}")
            raise
        
        # Convert back to polars and drop the original window_start column
        data = pl.from_pandas(df_pandas).drop("window_start")
        
        if data.shape[0] == 0:
            raise ValueError("Input data is empty")

        print(f"Processing {data.shape[0]} minute-level records for {date}")
        print("Aggregating to 30-minute bars...")
        
        # Aggregate minute data to 30-minute bars
        cleaned = (
            data
            .with_columns([
                # Create 30-minute time buckets
                (pl.col("timestamp").dt.truncate("30m")).alias("time_bucket")
            ])
            .group_by(["ticker", "time_bucket"])
            .agg([
                # OHLC aggregation
                pl.col("open").first().alias("stock30_open"),
                pl.col("high").max().alias("stock30_high"),
                pl.col("low").min().alias("stock30_low"),
                pl.col("close").last().alias("stock30_close"),
                pl.col("volume").sum().alias("stock30_volume"),
                pl.col("transactions").sum().alias("stock30_transactions")
            ])
            .with_columns([
                # Use time_bucket as timestamp
                pl.col("time_bucket").alias("timestamp"),
                
                # Extract date and time components
                pl.col("time_bucket").cast(pl.Date).alias("date"),
                pl.col("time_bucket").dt.hour().cast(pl.UInt8).alias("hour"),
                pl.col("time_bucket").dt.minute().cast(pl.UInt8).alias("minute"),
                
                # Flag last 30min of trading day (3:30 PM)
                (pl.col("time_bucket").dt.hour().eq(15) & 
                 pl.col("time_bucket").dt.minute().eq(30))
                .cast(pl.Boolean)
                .alias("stock30_is_last_30min")
            ])
            .sort(["ticker", "timestamp"])
            .with_columns([
                # Compute returns with outlier filtering
                (pl.col("stock30_close").pct_change().over("ticker"))
                .clip(-0.5, 0.5)  # Cap at ±50% to remove extreme outliers
                .alias("stock30_close_return"),
                
                # Rolling volatility with outlier-filtered returns
                (pl.col("stock30_close").pct_change().clip(-0.5, 0.5)
                 .rolling_std(window_size=5)
                 .over("ticker")).alias("stock30_rolling_vol_5")
            ])
            .drop("time_bucket")
        )
        
        # Convert to pandas for comprehensive dtype enforcement
        cleaned_final = cleaned.to_pandas()
        
        # Enforce feature dtypes to match finalized schema
        feature_dtypes = {
            'stock30_close_return': 'float64', 'stock30_rolling_vol_5': 'float64',
            'stock30_is_last_30min': 'boolean',
        }
        
        for col, dtype in feature_dtypes.items():
            if col in cleaned_final.columns:
                cleaned_final[col] = cleaned_final[col].astype(dtype)
        
        # Convert back to polars
        cleaned = pl.from_pandas(cleaned_final)

        # Validate outputs
        print(f"\nAggregated to {cleaned.shape[0]} 30-minute bars")
        print(f"Aggregation ratio: {data.shape[0] / cleaned.shape[0]:.2f} minutes per bar")
        print(f"Generated columns: {cleaned.columns}")
        
        print("\nNull counts:")
        for col in cleaned.columns:
            null_count = cleaned[col].null_count()
            null_pct = (null_count / cleaned.shape[0]) * 100
            print(f"{col}: {null_count} nulls ({null_pct:.2f}%)")

        # Basic stats
        print("\nValue ranges:")
        numeric_cols = [col for col in cleaned.columns 
                       if cleaned[col].dtype in [pl.Float64, pl.Int64]]
        for col in numeric_cols:
            try:
                stats = cleaned.select(col).describe()
                min_val = stats.filter(pl.col("statistic") == "min")[col].item()
                max_val = stats.filter(pl.col("statistic") == "max")[col].item()
                mean_val = stats.filter(pl.col("statistic") == "mean")[col].item()
                std_val = stats.filter(pl.col("statistic") == "std")[col].item()
                print(f"{col}:")
                if 'return' in col:
                    print(f"  min: {min_val:.6f} ({min_val*100:.2f}%)")
                    print(f"  max: {max_val:.6f} ({max_val*100:.2f}%)")
                    print(f"  mean: {mean_val:.6f} ({mean_val*100:.2f}%)")
                    print(f"  std: {std_val:.6f} ({std_val*100:.2f}%)")
                else:
                    print(f"  min: {min_val:.4f}")
                    print(f"  max: {max_val:.4f}")
                    print(f"  mean: {mean_val:.4f}")
                    print(f"  std: {std_val:.4f}")
            except Exception as e:
                print(f"{col}: Error getting stats - {e}")

        if not dry_run:
            # Write cleaned data
            cleaned.write_parquet(
                output_path,
                compression="zstd",
                statistics=True,
                use_pyarrow=True,
                pyarrow_options={"compression_level": 3}
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
                "input_rows": data.shape[0],
                "output_rows": cleaned.shape[0],
                "unique_tickers": cleaned["ticker"].n_unique(),
                "target_date": date,
                "timestamp_range": [timestamp_min, timestamp_max]
            }
        }

    except Exception as e:
        print(f"Error cleaning 30-minute stocks data: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean 30-minute stocks data")
    parser.add_argument("--date", type=str, required=True, help="Processing date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing files")
    
    args = parser.parse_args()
    result = run(args.date, dry_run=args.dry_run)
    print(f"Task result: {result}")

