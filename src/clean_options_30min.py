import polars as pl
from prefect import task
from prefect.tasks import task_input_hash
from datetime import timedelta
import os
from pathlib import Path
import argparse
from utils.profiling import profile_memory_and_time, profile_time
from utils.performance_utils import (
    compute_flow_signals_optimized,
    compute_gamma_signals_optimized,
    PERF_CONFIG
)

@task(
    name="clean_options_30min",
    description="Clean and transform 30-minute options data from parquet files",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    retries=3,
    retry_delay_seconds=60,
)
@profile_memory_and_time
def run(date: str, dry_run: bool = False, min_volume: int = 10, strikes_below: int = 5, strikes_above: int = 5, 
        vol_spike_multiplier: float = 2.0, flow_window: int = 1, gamma_squeeze_multiplier: float = 2.0) -> dict:
    """
    Clean and transform 30-minute options data using Polars streaming.
    
    Args:
        date: The processing date (for logging/tracking)
        dry_run: If True, simulate processing without writing files
    
    Returns:
        dict: Status information about the processing
    """
    try:
        print(f"Starting 30-minute options cleaning for date: {date}")
        
        # Input/output paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_path = os.path.join(project_root, "data/Parquet_data/option_minute")
        output_dir = os.path.join(project_root, "staged")
        output_path = os.path.join(output_dir, "options_30min_clean.parquet")
        
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        

        
        if not dry_run:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Define feature schema for final output
        feature_schema = {
            "opt30_strike": pl.Float64,
            "opt30_moneyness": pl.Float64,
            "opt30_mid_price_return": pl.Float64,
            "opt30_bid_ask_spread": pl.Float64,
            "opt30_implied_volatility": pl.Float64,
            "opt30_delta": pl.Float64,
            "opt30_theta": pl.Float64,
            "opt30_volume_return": pl.Float64,
            "opt30_rolling_vol_5": pl.Float64
        }
        
        # Load raw data with flexible schema (limit for memory efficiency)
        print("Loading data with flexible schema...")
        raw_df = pl.scan_parquet(input_path, schema=None, extra_columns='ignore').limit(50000).collect()
        
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
            if 'underlying' in df_pandas.columns:
                df_pandas['underlying'] = df_pandas['underlying'].astype(str)
        except Exception as e:
            print(f"Error during type conversion: {str(e)}")
            raise
        
        # Extract strike price from ticker (handle missing values)
        strike_match = df_pandas['ticker'].str.extract(r'(\d{8})', expand=False)
        df_pandas['strike'] = pd.to_numeric(strike_match, errors='coerce') / 1000.0
        df_pandas['strike'] = df_pandas['strike'].fillna(0.0)  # Fill missing strikes
        
        # Determine strike grid per underlying
        print(f"Applying strike grid selection ({strikes_below} below, {strikes_above} above ATM)...")
        filtered_rows = []
        
        for underlying in df_pandas['underlying'].unique():
            underlying_data = df_pandas[df_pandas['underlying'] == underlying]
            
            # Get unique strikes and sort them
            sorted_strikes = sorted(underlying_data['strike'].unique())
            
            # Estimate ATM strike (use median close price as proxy)
            underlying_price = underlying_data['close'].median()
            atm_index = min(range(len(sorted_strikes)), key=lambda i: abs(sorted_strikes[i] - underlying_price))
            
            # Select strike grid
            start_idx = max(0, atm_index - strikes_below)
            end_idx = atm_index + strikes_above + 1
            strike_grid = sorted_strikes[start_idx:end_idx]
            
            # Filter data to strike grid
            grid_data = underlying_data[underlying_data['strike'].isin(strike_grid)]
            filtered_rows.append(grid_data)
            
            print(f"{underlying}: ATM ~${underlying_price:.0f}, grid {len(strike_grid)} strikes (${min(strike_grid):.0f}-${max(strike_grid):.0f})")
        
        df_pandas = pd.concat(filtered_rows, ignore_index=True)
        print(f"Strike grid filtering: {len(df_pandas)} records retained")
        
        data = pl.from_pandas(df_pandas).drop("window_start")
        
        if data.shape[0] == 0:
            raise ValueError("Input data is empty")

        print(f"Processing {data.shape[0]} minute-level records for {date}")
        print("Aggregating to 30-minute bars...")
        
        # Aggregate minute data to 30-minute bars grouped by ticker AND strike
        cleaned = (
            data
            .with_columns([
                # Create 30-minute time buckets
                (pl.col("timestamp").dt.truncate("30m")).alias("time_bucket")
            ])
            .group_by(["ticker", "underlying", "strike", "time_bucket"])
            .agg([
                # OHLC aggregation
                pl.col("open").first().alias("opt30_open"),
                pl.col("high").max().alias("opt30_high"),
                pl.col("low").min().alias("opt30_low"),
                pl.col("close").last().alias("opt30_close"),
                pl.col("volume").sum().alias("opt30_volume"),
                pl.col("transactions").sum().alias("opt30_transactions")
            ])
            .with_columns([
                # Use time_bucket as timestamp
                pl.col("time_bucket").alias("timestamp"),
                
                # Extract date and time components
                pl.col("time_bucket").cast(pl.Date).alias("date"),
                pl.col("time_bucket").dt.hour().cast(pl.UInt8).alias("hour"),
                pl.col("time_bucket").dt.minute().cast(pl.UInt8).alias("minute"),
                
                # Calculate mid price and spread
                ((pl.col("opt30_high") + pl.col("opt30_low")) / 2).alias("opt30_mid_price"),
                ((pl.col("opt30_high") - pl.col("opt30_low")) / pl.col("opt30_close")).alias("opt30_bid_ask_spread"),
                
                # Basic Greeks estimates
                (pl.col("opt30_close") / 100.0).clip(0.1, 2.0).alias("opt30_implied_volatility"),
                pl.lit(0.4).cast(pl.Float64).alias("opt30_delta"),
                pl.lit(-0.01).cast(pl.Float64).alias("opt30_theta")
            ])
            .sort(["ticker", "timestamp"])
            .with_columns([
                # Compute returns with outlier filtering
                (pl.col("opt30_mid_price").pct_change().over(["ticker", "strike"]))
                .clip(-0.5, 0.5)
                .alias("opt30_mid_price_return"),
                
                # Volume returns
                (pl.col("opt30_volume").pct_change().over(["ticker", "strike"]))
                .clip(-0.5, 0.5)
                .alias("opt30_volume_return")
            ])
            .with_columns([
                # Optimized rolling calculations with native window functions
                (pl.col("opt30_mid_price_return")
                 .rolling_std(window_size=5, min_periods=1)
                 .over(["ticker", "strike"])).alias("opt30_rolling_vol_5"),
                
                # Volume mean for unusual volume detection
                (pl.col("opt30_volume")
                 .rolling_mean(window_size=5, min_periods=1)
                 .over(["ticker", "strike"])).alias("opt30_vol_mean_5")
            ])
            .with_columns([
                # Volume ratio and spike detection
                (pl.col("opt30_volume") / pl.col("opt30_vol_mean_5")).alias("opt30_vol_ratio"),
            ])
            .with_columns([
                # Volume spike boolean flag
                (pl.col("opt30_vol_ratio") > vol_spike_multiplier).alias("opt30_vol_spike")
            ])
            # Filter for liquid contracts only (configurable volume threshold)
            .filter(pl.col("opt30_volume") >= min_volume)
            .drop("time_bucket")
        )
        
        # Use optimized signal computation functions with config
        window_size = PERF_CONFIG['window_sizes']['flow_window']
        cleaned = compute_flow_signals_optimized(cleaned, window=window_size)
        cleaned = compute_gamma_signals_optimized(cleaned, window=window_size, gamma_squeeze_multiplier=gamma_squeeze_multiplier)
        
        # Native Polars dtype enforcement (no pandas conversion needed)
        cleaned = cleaned.with_columns([
            pl.col(col).cast(pl.Float64) for col in ['opt30_strike', 'opt30_moneyness', 'opt30_mid_price_return', 
                                                     'opt30_bid_ask_spread', 'opt30_implied_volatility', 'opt30_delta',
                                                     'opt30_vol_ratio', 'opt30_theta', 'opt30_volume_return', 'opt30_rolling_vol_5']
            if col in cleaned.columns
        ] + [
            pl.col('opt30_vol_spike').cast(pl.Boolean) if 'opt30_vol_spike' in cleaned.columns else pl.lit(None)
        ])

        # Validate outputs and log strike information
        unique_strikes = cleaned["strike"].n_unique()
        print(f"\nAggregated to {cleaned.shape[0]} 30-minute bars")
        print(f"Volume threshold: {min_volume}")
        print(f"Volume spike multiplier: {vol_spike_multiplier}x")
        print(f"Flow window: {flow_window}")
        print(f"Gamma squeeze multiplier: {gamma_squeeze_multiplier}x")
        
        # Log advanced signal summaries
        if 'opt30_flow_divergence' in cleaned.columns:
            flow_mean = cleaned['opt30_flow_divergence'].mean()
            print(f"Average flow divergence: {flow_mean:.2f}")
        
        if 'opt30_gamma_squeeze' in cleaned.columns:
            squeeze_count = cleaned['opt30_gamma_squeeze'].sum()
            squeeze_pct = (squeeze_count / cleaned.shape[0]) * 100
            print(f"Gamma squeezes detected: {squeeze_count}/{cleaned.shape[0]} ({squeeze_pct:.1f}%)")
        print(f"Unique strikes processed: {unique_strikes}")
        print(f"Aggregation ratio: {data.shape[0] / cleaned.shape[0]:.2f} minutes per bar")
        print(f"Generated columns: {cleaned.columns}")
        
        print("\nNull counts:")
        for col in cleaned.columns:
            null_count = cleaned[col].null_count()
            null_pct = (null_count / cleaned.shape[0]) * 100
            print(f"{col}: {null_count} nulls ({null_pct:.2f}%)")

        print("\nUnique counts:")
        print(f"Tickers: {cleaned['ticker'].n_unique()}")
        print(f"Strikes: {cleaned['strike'].n_unique()}")
        print(f"Underlying symbols: {cleaned['underlying'].n_unique()}")
        
        # Advanced signal statistics
        if 'opt30_call_flow' in cleaned.columns and 'opt30_put_flow' in cleaned.columns:
            call_flow_total = cleaned['opt30_call_flow'].sum()
            put_flow_total = cleaned['opt30_put_flow'].sum()
            print(f"Call/Put flow ratio: {call_flow_total / (put_flow_total + 1):.2f}")
        
        # Strike range analysis
        strike_stats = cleaned.select("strike").describe()
        min_strike = strike_stats.filter(pl.col("statistic") == "min")["strike"].item()
        max_strike = strike_stats.filter(pl.col("statistic") == "max")["strike"].item()
        print(f"Strike range: ${min_strike:.0f} - ${max_strike:.0f}")

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
                "unique_strikes": cleaned["strike"].n_unique(),
                "flow_window": flow_window,
                "gamma_squeeze_multiplier": gamma_squeeze_multiplier,
                "target_date": date,
                "timestamp_range": [timestamp_min, timestamp_max]
            }
        }

    except Exception as e:
        print(f"Error processing 30-minute options data: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean 30-minute options data")
    parser.add_argument("--date", type=str, required=True, help="Processing date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing files")
    parser.add_argument("--min-volume", type=int, default=10, help="Minimum volume threshold (default: 10)")
    parser.add_argument("--strikes-below", type=int, default=5, help="Number of strikes below ATM (default: 5)")
    parser.add_argument("--strikes-above", type=int, default=5, help="Number of strikes above ATM (default: 5)")
    parser.add_argument("--vol-spike-multiplier", type=float, default=2.0, help="Volume spike threshold multiplier (default: 2.0)")
    parser.add_argument("--flow-window", type=int, default=1, help="Flow divergence smoothing window (default: 1)")
    parser.add_argument("--gamma-squeeze-multiplier", type=float, default=2.0, help="Gamma squeeze threshold multiplier (default: 2.0)")
    
    args = parser.parse_args()
    result = run(args.date, dry_run=args.dry_run, min_volume=args.min_volume, 
                strikes_below=args.strikes_below, strikes_above=args.strikes_above,
                vol_spike_multiplier=args.vol_spike_multiplier, flow_window=args.flow_window,
                gamma_squeeze_multiplier=args.gamma_squeeze_multiplier)
    print(f"Task result: {result}")
