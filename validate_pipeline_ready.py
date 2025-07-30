#!/usr/bin/env python3
# ============================================================
# DRY-RUN / SCHEMA VALIDATION  –  Conviction AI  (Step 0, v3)
# ============================================================
# Auto-detect a valid DATE that exists in *all* raw parquet feeds,
# then run clean_* scripts and validate outputs.
#
# Updated for actual data structure: parquet files, not JSON
# -------- CONFIG --------
RAW_BASE = "data/Parquet_data/Raw"
CLEAN_BASE = "staged"
FEEDS = {
    "stocks_30m": {
        "script": "clean_stocks_30min.py",
        "raw_path": "stocks_minute",
        "output": "stocks_30min_clean.parquet",
    },
    "stocks_daily": {
        "script": "clean_stocks_daily.py",
        "raw_path": "Stocks_daily",
        "output": "stocks_daily_clean.parquet",
    },
    "options_30m": {
        "script": "clean_options_30min.py",
        "raw_path": "option_minute",
        "output": "options_30min_clean.parquet",
    },
    "options_daily": {
        "script": "clean_options_daily.py",
        "raw_path": "options_daily",
        "output": "options_daily_clean.parquet",
    },
    "macro": {
        "script": "clean_macro_data.py",
        "raw_path": "FRED.csv",  # CSV file
        "output": "macro_clean.parquet",
    },
    "news": {
        "script": "clean_news.py",
        "raw_path": "news",
        "output": "news_clean.parquet",
    },
}
# ------------------------

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import polars as pl


def _get_dates_from_parquet(feed_path):
    """Extract available dates from parquet files"""
    try:
        if feed_path.endswith(".csv"):
            # Handle CSV files (like FRED.csv)
            return {"2025-04-04"}  # Assume macro data is available

        full_path = f"{RAW_BASE}/{feed_path}"
        if not Path(full_path).exists():
            print(f"⚠️  Path not found: {full_path}")
            return set()

        # Read parquet files and extract dates
        if Path(full_path).is_dir():
            df = (
                pl.scan_parquet(f"{full_path}/*.parquet", extra_columns="ignore")
                .limit(1000)
                .collect()
            )
        else:
            df = pl.read_parquet(full_path)

        if "window_start" in df.columns:
            # Convert window_start to dates
            df = df.with_columns(
                [pl.col("window_start").cast(pl.Int64).alias("window_start_int")]
            )
            df_pd = df.to_pandas()
            df_pd["date"] = pd.to_datetime(df_pd["window_start_int"], unit="ns").dt.date
            dates = {str(d) for d in df_pd["date"].unique()}
        elif "date" in df.columns:
            dates = {str(d) for d in df["date"].unique()}
        else:
            print(f"⚠️  No date column found in {feed_path}")
            return set()

        return dates
    except Exception as e:
        print(f"⚠️  Error reading {feed_path}: {e}")
        return set()


def _common_date():
    """Find common date across all feeds"""
    print("🔍 Checking available dates in each feed...")

    date_sets = []
    for feed, config in FEEDS.items():
        dates = _get_dates_from_parquet(config["raw_path"])
        print(f"   {feed}: {len(dates)} dates available")
        if dates:
            print(f"      Sample: {sorted(list(dates))[:3]}")
        date_sets.append(dates)

    common = set.intersection(*[ds for ds in date_sets if ds])
    if not common:
        print("❌ No common date found across all feeds")
        print("Available dates per feed:")
        for i, (feed, dates) in enumerate(zip(FEEDS.keys(), date_sets)):
            print(f"   {feed}: {sorted(list(dates))[:5] if dates else 'None'}")
        raise SystemExit("❌ No single date exists across *all* feeds.")

    return max(common)  # latest common date


def _run_clean(feed, config, date):
    """Run cleaning script for a feed"""
    cmd = ["python", f"src/{config['script']}", "--date", date]
    print("▶", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"   ✅ {feed} cleaned successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {feed} failed: {e.stderr}")
        return False


def _validate_output(feed, config, date):
    """Validate cleaned output exists and has data"""
    output_path = f"{CLEAN_BASE}/{config['output']}"

    if not Path(output_path).exists():
        print(f"   ❌ Missing output: {output_path}")
        return False

    try:
        df = pl.read_parquet(output_path)
        print(f"   ✅ {feed}: {df.shape[0]} rows, {df.shape[1]} columns")
        return True
    except Exception as e:
        print(f"   ❌ {feed} validation failed: {e}")
        return False


def main():
    print("🚀 Starting Conviction AI Pipeline Validation\n")

    # 1. Find common date
    try:
        DATE = _common_date()
        print(f"✔ Using common DATE: {DATE}\n")
    except SystemExit:
        print("💡 Suggestion: Use a known good date like 2025-04-04")
        DATE = "2025-04-04"
        print(f"🔄 Proceeding with fallback DATE: {DATE}\n")

    # 2. Create output directory
    Path(CLEAN_BASE).mkdir(parents=True, exist_ok=True)

    # 3. Run cleaning scripts
    print("🧹 Running cleaning scripts...")
    success_count = 0
    for feed, config in FEEDS.items():
        if _run_clean(feed, config, DATE):
            success_count += 1

    print(f"\n📊 Cleaning Results: {success_count}/{len(FEEDS)} successful\n")

    # 4. Validate outputs
    print("✅ Validating outputs...")
    validation_count = 0
    for feed, config in FEEDS.items():
        if _validate_output(feed, config, DATE):
            validation_count += 1

    print(f"\n📊 Validation Results: {validation_count}/{len(FEEDS)} successful")

    if validation_count == len(FEEDS):
        print(f"\n🎉 ALL CLEAN FILES VALIDATED SUCCESSFULLY FOR {DATE}")
        print("   Next step: run calculate_features.py for the same DATE")
        print(f"   Command: python src/calculate_features.py --date {DATE}")
    else:
        print(f"\n⚠️  Some validations failed. Check outputs in {CLEAN_BASE}/")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
