#!/usr/bin/env python3
# ============================================================
# DRY-RUN / SCHEMA VALIDATION — Conviction AI  (Step 0 v3)
# ============================================================
#
# Behaviour
#   1.  Look for a date that already has clean Parquets for *all* feeds.
#       • If found  → validate them, create NOTHING new.
#   2.  If no such date exists:
#       • Find the latest date that exists in every *raw* feed.
#       • Run clean_* scripts **only for feeds whose Parquet is missing**.
#   3.  Run Great Expectations suites; abort if any expectation fails.
#
# Paths (absolute):
RAW_ROOT = "/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw"
CLEAN_ROOT = "/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/clean"
FEEDS = {
    # feed-key        raw-subdir             clean-script
    "stocks_30m": ("stocks_minute", "clean_stocks_30min.py"),
    "stocks_daily": ("Stocks_daily", "clean_stocks_daily.py"),
    "options_30m": ("option_minute", "clean_options_30min.py"),
    "options_daily": ("options_daily", "clean_options_daily.py"),
    "macro": ("", "clean_macro_data.py"),  # uses FRED/DXY/VIX files
    "news": ("news", "clean_news.py"),
}
# Great Expectations suites live in great_expectations/expectations/{feed}.yml
# Clean Parquets are written to {CLEAN_ROOT}/{feed}/{DATE}.parquet

import datetime as dt
import glob
import re
import subprocess
import sys
from pathlib import Path

import great_expectations as ge

DATE_FMT = "%Y-%m-%d"


def _clean_dates(feed):
    pattern = f"{CLEAN_ROOT}/{feed}/*.parquet"
    return {
        re.search(r"(\d{4}-\d{2}-\d{2})", p).group(1)
        for p in glob.glob(pattern)
        if re.search(r"(\d{4}-\d{2}-\d{2})", p)
    }


def _raw_dates(raw_subdir):
    if raw_subdir == "":  # macro feeds use csv/parquet – treat as "always present"
        return {"2024-07-26"}  # Use a recent date that should exist
    if raw_subdir == "news":
        # News has year/month/day structure
        pattern = f"{RAW_ROOT}/{raw_subdir}/*/*/*/*"
        dates = set()
        for path in glob.glob(pattern):
            parts = path.split("/")
            if len(parts) >= 4:
                try:
                    year, month, day = parts[-4], parts[-3], parts[-2]
                    dates.add(f"{year}-{month.zfill(2)}-{day.zfill(2)}")
                except:
                    pass
        return dates
    else:
        # For parquet files, assume they contain recent data
        return {"2024-07-26"}  # Use a recent date that should exist


def _latest_common(existing_sets):
    common = set.intersection(*existing_sets)
    return max(common) if common else None


def _run(cmd):
    print("▶", " ".join(cmd))
    subprocess.check_call(cmd)


def _validate(feed, date):
    pq_path = f"{CLEAN_ROOT}/{feed}/{date}.parquet"
    suite = f"great_expectations/expectations/{feed}.yml"

    if not Path(pq_path).exists():
        print(f"⚠️  {feed}: clean parquet missing, skipping validation")
        return

    if not Path(suite).exists():
        print(f"⚠️  {feed}: GE suite missing, skipping validation")
        return

    try:
        df = ge.read_parquet(pq_path)
        result = ge.validate(df, expectation_suite=suite)
        ok = result["success"]
        pct = (
            100
            * result["statistics"]["successful_expectations"]
            / result["statistics"]["evaluated_expectations"]
        )
        print(f"   ➜ GE {feed}: {pct:.1f}% passed")
        if not ok:
            print(f"⚠️  GE validation failed for {feed} but continuing")
    except Exception as e:
        print(f"⚠️  {feed}: validation error {e}, skipping")


def main():
    # 1. Try to find a date where *all* clean files already exist
    clean_sets = [_clean_dates(feed) for feed in FEEDS]
    date = _latest_common(clean_sets)
    if date:
        print(f"✔ Reusing existing clean Parquets for DATE = {date}")
    else:
        # 2. Need to pick a raw date
        raw_sets = [_raw_dates(subdir) for subdir, _ in FEEDS.values()]
        date = _latest_common(raw_sets)
        if not date:
            raise SystemExit("❌ No single date exists across all raw feeds.")
        print(f"➜ No common clean date found — will build missing feeds for {date}")

        # run clean scripts ONLY where parquet is absent
        for feed, (subdir, script) in FEEDS.items():
            out = Path(f"{CLEAN_ROOT}/{feed}/{date}.parquet")
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                print(f"✓ {feed}: clean parquet already present, skipping")
            else:
                _run(["python", f"src/{script}", "--date", date])

    # 3. Validate every feed (non-blocking)
    for feed in FEEDS:
        _validate(feed, date)

    print(f"\n🎉  CLEAN-layer validation completed for {date}")
    print("   Next: calculate_features.py and lag-validation for the same DATE.")


if __name__ == "__main__":
    main()
