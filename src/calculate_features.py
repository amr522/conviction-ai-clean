#!/usr/bin/env python3
"""
Feature calculation module using Polars for high-performance data processing.
Generates final feature matrix from daily and intraday master datasets.
"""

import argparse
import polars as pl
from datetime import datetime
import concurrent.futures
from pathlib import Path
from utils.lineage_utils import LineageTracker


def parse_date_range(date_str):
    """Parse date string - single date or range (YYYY-MM-DD,YYYY-MM-DD)"""
    if ',' in date_str:
        start_str, end_str = date_str.split(',')
        return datetime.strptime(start_str.strip(), '%Y-%m-%d').date(), \
               datetime.strptime(end_str.strip(), '%Y-%m-%d').date()
    else:
        single_date = datetime.strptime(date_str.strip(), '%Y-%m-%d').date()
        return single_date, single_date


def calculate_rolling_features(df, window):
    """Calculate rolling features for a single ticker group"""
    return df.sort("date").with_columns([
        # Macro rolling features
        pl.col("fred_fed_funds_rate").rolling_mean(window).alias("fred_rate_mean"),
        pl.col("vix_index").rolling_std(window).alias("vix_std"),
        pl.col("news_count").rolling_sum(window).alias("news_count_rolling"),
        
        # Options rolling features
        pl.col("optd_iv30").rolling_mean(window).alias("optd_iv30_mean"),
        pl.col("optd_volume").rolling_std(window).alias("optd_volume_std"),
        
        # Stock rolling features
        pl.col("stockd_return_1d").rolling_std(window).alias("stockd_vol_rolling"),
        pl.col("stockd_volume").rolling_mean(window).alias("stockd_volume_mean")
    ])


def process_ticker_chunk(ticker_chunk, dm, window):
    """Process a chunk of tickers in parallel"""
    chunk_data = dm.filter(pl.col("ticker").is_in(ticker_chunk))
    return chunk_data.group_by("ticker").map_groups(
        lambda df: calculate_rolling_features(df, window)
    )


def calculate_intraday_features(im):
    """Calculate intraday features including lagged returns and spikes"""
    # Pivot intraday data to wide format for return calculations
    pivot = im.select(["ticker", "timestamp", "opt30_mid_price"]) \
              .pivot(values="opt30_mid_price", index="timestamp", columns="ticker")
    
    # Calculate 1-hour returns (2 × 30min periods)
    ret_cols = [col for col in pivot.columns if col != "timestamp"]
    ret_exprs = [
        (pl.col(col).pct_change(2)).alias(f"{col}_ret_1h") 
        for col in ret_cols
    ]
    
    pivot_ret = pivot.with_columns(ret_exprs)
    
    # Melt back to long format
    ret_1h = pivot_ret.select(["timestamp"] + [f"{col}_ret_1h" for col in ret_cols]) \
                     .melt(id_vars="timestamp", variable_name="ticker_ret", value_name="ret_1h") \
                     .with_columns(
                         pl.col("ticker_ret").str.replace("_ret_1h", "").alias("ticker")
                     ).select(["timestamp", "ticker", "ret_1h"])
    
    return ret_1h


def calculate_cross_sectional_features(features):
    """Calculate cross-sectional z-scores and relative features"""
    return features.with_columns([
        # Volume z-score across tickers at each timestamp
        ((pl.col("optd_volume") - pl.col("optd_volume").mean().over("date")) /
         pl.col("optd_volume").std().over("date")).alias("vol_zscore"),
        
        # IV percentile across tickers
        pl.col("optd_iv30").rank().over("date").alias("iv_rank"),
        
        # Return relative to market
        (pl.col("stockd_return_1d") - pl.col("stockd_return_1d").mean().over("date")).alias("ret_relative")
    ])


def main():
    parser = argparse.ArgumentParser(description="Calculate features using Polars")
    parser.add_argument("--daily-master-path", required=True, help="Path to daily master parquet")
    parser.add_argument("--intraday-master-path", required=True, help="Path to intraday master parquet")
    parser.add_argument("--output-path", required=True, help="Output path for features")
    parser.add_argument("--date", required=True, help="Date or date range (YYYY-MM-DD or YYYY-MM-DD,YYYY-MM-DD)")
    parser.add_argument("--window-days", type=int, default=30, help="Rolling window size")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")
    
    args = parser.parse_args()
    
    # Configure Polars for GPU if requested
    if args.use_gpu:
        try:
            pl.Config.set_streaming_chunk_size(10000)
        except:
            print("GPU acceleration not available, using CPU")
    
    print(f"Loading data from {args.daily_master_path} and {args.intraday_master_path}")
    
    # Initialize lineage tracking
    lineage = LineageTracker()
    lineage.start_run(
        "calculate_features",
        inputs=[args.daily_master_path, args.intraday_master_path],
        outputs=[args.output_path]
    )
    
    try:
        # Load data
        dm = pl.read_parquet(args.daily_master_path)
        im = pl.read_parquet(args.intraday_master_path)
    
    print(f"Loaded {len(dm)} daily records and {len(im)} intraday records")
    
    # Parse date range
    start_date, end_date = parse_date_range(args.date)
    print(f"Processing date range: {start_date} to {end_date}")
    
    # Filter data by date range
    dm_filtered = dm.filter(
        (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
    )
    im_filtered = im.filter(
        (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
    )
    
    # Calculate rolling features
    print("Calculating rolling features...")
    window = args.window_days
    
    if args.n_jobs > 1:
        # Parallel processing
        tickers = dm_filtered["ticker"].unique().to_list()
        chunk_size = len(tickers) // args.n_jobs + 1
        ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = [
                executor.submit(process_ticker_chunk, chunk, dm_filtered, window)
                for chunk in ticker_chunks
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Combine results
        roll_features = pl.concat(results)
    else:
        # Sequential processing
        roll_features = dm_filtered.group_by("ticker").map_groups(
            lambda df: calculate_rolling_features(df, window)
        )
    
    print("Calculating intraday features...")
    # Calculate intraday features
    intraday_features = calculate_intraday_features(im_filtered)
    
    print("Joining features...")
    # Join daily and intraday features
    features = roll_features.join(
        intraday_features, 
        on=["ticker"], 
        how="left"
    )
    
    print("Calculating cross-sectional features...")
    # Calculate cross-sectional features
    features = calculate_cross_sectional_features(features)
    
    # Ensure output directory exists
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
        print(f"Writing {len(features)} feature records to {args.output_path}")
        # Write output
        features.write_parquet(args.output_path)
        
        print("Feature calculation completed successfully!")
        print(f"Output columns: {features.columns}")
        
        # Complete lineage tracking
        lineage.complete_run(success=True)
        
    except Exception as e:
        lineage.complete_run(success=False)
        raise e


if __name__ == "__main__":
    main()