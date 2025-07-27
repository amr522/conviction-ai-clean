#!/usr/bin/env python3
"""
Investigate output files created by cleaning scripts
"""
import os
from pathlib import Path
import pyarrow.parquet as pq

def check_file_exists(path: str, description: str):
    """Check if file exists and get basic info"""
    p = Path(path)
    if p.exists():
        try:
            if p.suffix == '.parquet':
                tbl = pq.read_table(p)
                print(f"✅ {description}: {p}")
                print(f"   • Rows: {len(tbl):,}")
                print(f"   • Columns: {len(tbl.column_names)}")
                print(f"   • Size: {p.stat().st_size / 1024 / 1024:.2f} MB")
            else:
                print(f"✅ {description}: {p}")
                print(f"   • Size: {p.stat().st_size / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"⚠️  {description}: {p} (Error reading: {e})")
    else:
        print(f"❌ {description}: {p} (NOT FOUND)")

def main():
    print("🔍 Investigating output files from cleaning scripts...\n")
    
    # Base paths
    base_path = "/Users/amroheidak/Desktop/conviction-ai-clean"
    staged_dir = f"{base_path}/staged"
    data_dir = f"{base_path}/data/Parquet_data"
    
    print("=== MACRO DATA OUTPUTS (clean_macro_data.py) ===")
    check_file_exists(f"{data_dir}/fred.parquet", "FRED data")
    check_file_exists(f"{data_dir}/vix_data.parquet", "VIX data") 
    check_file_exists(f"{data_dir}/dxy.parquet", "DXY data")
    check_file_exists(f"{data_dir}/news_data.parquet", "News data")
    
    print("\n=== STAGED OUTPUTS (cleaning scripts) ===")
    check_file_exists(f"{staged_dir}/options_30min_clean.parquet", "Options 30min cleaned")
    check_file_exists(f"{staged_dir}/options_daily_clean.parquet", "Options daily cleaned")
    check_file_exists(f"{staged_dir}/stocks_30min_clean.parquet", "Stocks 30min cleaned")
    check_file_exists(f"{staged_dir}/stocks_daily_clean.parquet", "Stocks daily cleaned")
    
    print("\n=== STAGED DIRECTORY CONTENTS ===")
    staged_path = Path(staged_dir)
    if staged_path.exists():
        for file in staged_path.glob("*"):
            if file.is_file():
                size_mb = file.stat().st_size / 1024 / 1024
                print(f"📄 {file.name} ({size_mb:.2f} MB)")
    else:
        print("❌ Staged directory does not exist")
    
    print("\n=== RAW INPUT SOURCES ===")
    # Check raw inputs that scripts expect
    raw_inputs = [
        (f"{base_path}/data/Parquet_data/option_minute", "Options minute data"),
        (f"{base_path}/raw/options_daily.parquet", "Options daily raw"),
        (f"{base_path}/data/Parquet_data/stocks_minute", "Stocks minute data"),
        (f"{base_path}/data/Parquet_data/Stocks_daily", "Stocks daily directory"),
        (f"{base_path}/data/Parquet_data/Raw/FRED.csv", "FRED raw CSV"),
        (f"{base_path}/data/Parquet_data/Raw/DXY.csv", "DXY raw CSV"),
        (f"{base_path}/data/Parquet_data/Raw/vix_data.json", "VIX raw JSON"),
        (f"{base_path}/data/Parquet_data/Raw/news", "News raw directory"),
    ]
    
    for path, desc in raw_inputs:
        p = Path(path)
        if p.exists():
            if p.is_dir():
                file_count = len(list(p.glob("*")))
                print(f"✅ {desc}: {p} ({file_count} files)")
            else:
                size_mb = p.stat().st_size / 1024 / 1024
                print(f"✅ {desc}: {p} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {desc}: {p} (NOT FOUND)")

if __name__ == "__main__":
    main()