#!/usr/bin/env python3
"""Distributed historical backfill flow using Prefect and Dask."""

from datetime import datetime, timedelta
from typing import List

from prefect import flow, task
from prefect_dask import DaskTaskRunner
import polars as pl


@task
def process_date_chunk(date_str: str, tickers: List[str]) -> dict:
    """Process a single date chunk with given tickers."""
    print(f"Processing {date_str} for {len(tickers)} tickers")
    
    # Simulate data processing
    results = {
        "date": date_str,
        "tickers_processed": len(tickers),
        "features_generated": len(tickers) * 100,  # Mock feature count
        "status": "completed"
    }
    
    return results


@task
def aggregate_results(results: List[dict]) -> dict:
    """Aggregate results from all date chunks."""
    total_tickers = sum(r["tickers_processed"] for r in results)
    total_features = sum(r["features_generated"] for r in results)
    
    return {
        "total_dates": len(results),
        "total_tickers_processed": total_tickers,
        "total_features_generated": total_features,
        "status": "aggregation_complete"
    }


@flow(task_runner=DaskTaskRunner(address="tcp://127.0.0.1:8786"))
def distributed_backfill_flow(
    start_date: str,
    end_date: str,
    tickers: List[str] = None
) -> dict:
    """Distributed backfill flow using Dask for parallel processing."""
    
    if tickers is None:
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]  # Default tickers
    
    print(f"🚀 Starting distributed backfill from {start_date} to {end_date}")
    print(f"📊 Processing {len(tickers)} tickers")
    
    # Generate date range
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    print(f"📅 Processing {len(dates)} dates")
    
    # Process each date in parallel using Dask
    futures = []
    for date_str in dates:
        future = process_date_chunk.submit(date_str, tickers)
        futures.append(future)
    
    # Wait for all tasks to complete
    results = [future.result() for future in futures]
    
    # Aggregate final results
    final_result = aggregate_results(results)
    
    print("✅ Distributed backfill completed successfully")
    return final_result


@flow
def simple_backfill_flow(date: str) -> dict:
    """Simple backfill flow for single date processing."""
    print(f"🔄 Processing single date: {date}")
    
    result = process_date_chunk(date, ["AAPL", "GOOGL", "MSFT"])
    
    print("✅ Single date backfill completed")
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run historical backfill flow")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--distributed", action="store_true", help="Use distributed processing")
    parser.add_argument("--tickers", nargs="+", help="List of tickers to process")
    
    args = parser.parse_args()
    
    if args.distributed and args.end_date:
        # Run distributed backfill
        result = distributed_backfill_flow(
            args.start_date,
            args.end_date,
            args.tickers
        )
    else:
        # Run simple backfill for single date
        result = simple_backfill_flow(args.start_date)
    
    print(f"📋 Final result: {result}")