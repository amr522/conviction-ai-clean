#!/usr/bin/env python3

import polars as pl
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_daily_master():
    """Generate a minimal daily master dataset for testing."""
    n_days = 10
    base_date = datetime(2025, 1, 1)
    dates = [(base_date - timedelta(days=x)).date() for x in range(n_days)]
    
    data = {
        'date': dates,
        'symbol': ['SPY'] * n_days,
        'close': np.random.uniform(400, 500, n_days),
        'volume': np.random.uniform(1e6, 5e6, n_days),
        'high': np.random.uniform(400, 500, n_days),
        'low': np.random.uniform(400, 500, n_days),
        'open': np.random.uniform(400, 500, n_days)
    }
    
    return pl.DataFrame(data)

def generate_sample_intraday_master():
    """Generate a minimal intraday master dataset for testing."""
    n_days = 10
    intervals_per_day = 13  # 30-minute intervals in a trading day
    base_date = datetime(2025, 1, 1)
    
    dates = []
    times = []
    for d in range(n_days):
        current_date = base_date - timedelta(days=d)
        for i in range(intervals_per_day):
            dates.append(current_date.date())
            times.append(f"{9+i//2:02d}:{(i%2)*30:02d}")
    
    n_records = len(dates)
    data = {
        'date': dates,
        'time': times,
        'symbol': ['SPY'] * n_records,
        'close': np.random.uniform(400, 500, n_records),
        'volume': np.random.uniform(1e5, 5e5, n_records),
        'high': np.random.uniform(400, 500, n_records),
        'low': np.random.uniform(400, 500, n_records),
        'open': np.random.uniform(400, 500, n_records)
    }
    
    return pl.DataFrame(data)

def main():
    # Create output directory if it doesn't exist
    output_dir = "data/Parquet_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save daily master
    daily_df = generate_sample_daily_master()
    daily_df.write_parquet(f"{output_dir}/daily_master.parquet")
    
    # Generate and save intraday master
    intraday_df = generate_sample_intraday_master()
    intraday_df.write_parquet(f"{output_dir}/intraday_master.parquet")

if __name__ == "__main__":
    main()
