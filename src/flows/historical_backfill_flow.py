#!/usr/bin/env python3
"""
Prefect flow for parallel historical backfill execution.
"""

import subprocess
from datetime import date, timedelta
from typing import List

from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner


@task(retries=3, retry_delay_seconds=60)
def run_pipeline_for_date(dt: str, raw_fred_csv: str = None, raw_vix_json: str = None, 
                         raw_dxy_csv: str = None, raw_news_dir: str = None) -> dict:
    """
    Run the full pipeline for a specific date with retry logic.
    
    Args:
        dt: Date string in YYYY-MM-DD format
        
    Returns:
        Dict with execution results
    """
    cmd = f"python src/run_full_pipeline.py --date {dt} --check-schema"
    if raw_fred_csv:
        cmd += f" --raw-fred-csv {raw_fred_csv}"
    if raw_vix_json:
        cmd += f" --raw-vix-json {raw_vix_json}"
    if raw_dxy_csv:
        cmd += f" --raw-dxy-csv {raw_dxy_csv}"
    if raw_news_dir:
        cmd += f" --raw-news-dir {raw_news_dir}"
    
    try:
        result = subprocess.run(
            cmd.split(),
            capture_output=True,
            text=True,
            check=True,
            timeout=3600  # 1 hour timeout per date
        )
        
        return {
            "date": dt,
            "status": "success",
            "stdout": result.stdout[-1000:],  # Last 1000 chars
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "returncode": result.returncode
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "date": dt,
            "status": "failed",
            "stdout": e.stdout[-1000:] if e.stdout else "",
            "stderr": e.stderr[-1000:] if e.stderr else "",
            "returncode": e.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "date": dt,
            "status": "timeout",
            "stdout": "",
            "stderr": "Pipeline execution timed out after 1 hour",
            "returncode": -1
        }


@flow(
    name="Historical Backfill",
    log_prints=True,
    task_runner=ConcurrentTaskRunner()
)
def backfill_flow(start_date: str, end_date: str, max_workers: int = 24,
                  raw_fred_csv: str = None, raw_vix_json: str = None,
                  raw_dxy_csv: str = None, raw_news_dir: str = None) -> List[dict]:
    """
    Execute historical backfill across date range in parallel.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_workers: Maximum parallel workers (default: 24 for M2 Ultra)
        
    Returns:
        List of execution results for each date
    """
    # Generate date list
    sd = date.fromisoformat(start_date)
    ed = date.fromisoformat(end_date)
    dates = [(sd + timedelta(days=i)).isoformat() for i in range((ed - sd).days + 1)]
    
    print(f"Starting backfill for {len(dates)} dates: {start_date} to {end_date}")
    print(f"Using {max_workers} parallel workers")
    
    # Run pipeline for each date in parallel
    results = run_pipeline_for_date.map(
        dates, 
        raw_fred_csv=[raw_fred_csv] * len(dates),
        raw_vix_json=[raw_vix_json] * len(dates),
        raw_dxy_csv=[raw_dxy_csv] * len(dates),
        raw_news_dir=[raw_news_dir] * len(dates)
    )
    
    # Collect and summarize results
    completed_results = []
    success_count = 0
    failed_count = 0
    
    for result in results:
        completed_results.append(result)
        if result["status"] == "success":
            success_count += 1
        else:
            failed_count += 1
            print(f"❌ Failed: {result['date']} - {result['stderr'][:200]}")
    
    print(f"\n📊 Backfill Summary:")
    print(f"  ✅ Successful: {success_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  📈 Success Rate: {success_count/len(dates)*100:.1f}%")
    
    return completed_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Historical backfill with Prefect")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-workers", type=int, default=24, help="Max parallel workers")
    parser.add_argument("--raw-fred-csv", help="Path to raw FRED CSV")
    parser.add_argument("--raw-vix-json", help="Path to raw VIX JSON")
    parser.add_argument("--raw-dxy-csv", help="Path to raw DXY CSV")
    parser.add_argument("--raw-news-dir", help="Path to raw news directory")
    
    args = parser.parse_args()
    
    results = backfill_flow(
        args.start_date, args.end_date, args.max_workers,
        args.raw_fred_csv, args.raw_vix_json, args.raw_dxy_csv, args.raw_news_dir
    )