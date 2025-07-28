#!/usr/bin/env python3
"""
Historical Backfill - Use existing run_full_pipeline.py for date range
Following ML_TRAINING_ROADMAP.md Day 1 requirements
"""

import argparse
import subprocess
import sys
from datetime import datetime, timedelta


def run_pipeline_for_date(date_str):
    """Run the full pipeline for a single date"""
    cmd = [sys.executable, "src/run_full_pipeline.py", "--date", date_str]

    print(f"🚀 Processing {date_str}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {date_str} completed successfully")
        return True
    else:
        print(f"❌ {date_str} failed: {result.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run historical backfill")
    parser.add_argument(
        "--start-date", default="2018-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument("--skip-weekends", action="store_true", help="Skip weekends")

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    current_date = start_date
    success_count = 0
    total_count = 0

    print(f"📅 Running backfill from {start_date.date()} to {end_date.date()}")

    while current_date <= end_date:
        # Skip weekends if requested
        if args.skip_weekends and current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        date_str = current_date.strftime("%Y-%m-%d")
        total_count += 1

        if run_pipeline_for_date(date_str):
            success_count += 1

        current_date += timedelta(days=1)

    print(f"\n🎉 Backfill completed: {success_count}/{total_count} dates successful")


if __name__ == "__main__":
    main()
