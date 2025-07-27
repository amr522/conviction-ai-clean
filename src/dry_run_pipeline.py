#!/usr/bin/env python3
"""
Dry run pipeline harness for testing and validation.
"""

import argparse
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_pipeline import run_full_pipeline
from validate_schemas import validate_all_schemas


def dry_run_harness(date: str, check_schema: bool = False, flow_window: int = 1,
                   gamma_squeeze_multiplier: float = 2.0, daily_vol_spike_multiplier: float = 2.0,
                   use_raw_macro: bool = False, raw_fred_csv: str = None, raw_vix_json: str = None,
                   raw_dxy_csv: str = None, raw_news_dir: str = None):
    """
    Run pipeline in dry-run mode with optional schema validation.
    
    Args:
        date: Processing date (YYYY-MM-DD)
        check_schema: Whether to validate schemas after processing
        flow_window: Flow divergence smoothing window
        gamma_squeeze_multiplier: Gamma squeeze threshold multiplier
        daily_vol_spike_multiplier: Daily volume spike threshold multiplier
    """
    print(f"=== DRY RUN PIPELINE HARNESS FOR {date} ===")
    
    # Run pipeline in dry-run mode
    result = run_full_pipeline(
        date=date,
        dry_run=True,
        flow_window=flow_window,
        gamma_squeeze_multiplier=gamma_squeeze_multiplier,
        daily_vol_spike_multiplier=daily_vol_spike_multiplier,
        check_schema=False,  # Schema validation doesn't work in dry-run mode
        use_raw_macro=use_raw_macro,
        raw_fred_csv=raw_fred_csv,
        raw_vix_json=raw_vix_json,
        raw_dxy_csv=raw_dxy_csv,
        raw_news_dir=raw_news_dir
    )
    
    # If schema validation is requested and we have existing files
    if check_schema:
        print("\n" + "="*50)
        print("SCHEMA VALIDATION: Checking existing Parquet files...")
        
        try:
            validate_all_schemas("staged", "schemas/option_parquet_schema.json")
        except Exception as e:
            print(f"⚠️  Schema validation failed: {e}")
            result['schema_validation'] = {'status': 'failed', 'error': str(e)}
        else:
            result['schema_validation'] = {'status': 'passed'}
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dry run pipeline harness with schema validation")
    parser.add_argument("--date", type=str, required=True, help="Processing date (YYYY-MM-DD)")
    parser.add_argument("--check-schema", action="store_true", help="Validate existing Parquet schemas")
    parser.add_argument("--flow-window", type=int, default=1, help="Flow divergence smoothing window (default: 1)")
    parser.add_argument("--gamma-squeeze-multiplier", type=float, default=2.0, help="Gamma squeeze threshold multiplier (default: 2.0)")
    parser.add_argument("--daily-vol-spike-multiplier", type=float, default=2.0, help="Daily volume spike threshold multiplier (default: 2.0)")
    parser.add_argument("--use-raw-macro", action="store_true", help="Force use of raw macro data sources")
    parser.add_argument("--raw-fred-csv", default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/FRED.csv")
    parser.add_argument("--raw-vix-json", default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/vix_data.json")
    parser.add_argument("--raw-dxy-csv", default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/DXY.csv")
    parser.add_argument("--raw-news-dir", default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/news")
    
    args = parser.parse_args()
    
    result = dry_run_harness(
        date=args.date,
        check_schema=args.check_schema,
        flow_window=args.flow_window,
        gamma_squeeze_multiplier=args.gamma_squeeze_multiplier,
        daily_vol_spike_multiplier=args.daily_vol_spike_multiplier,
        use_raw_macro=args.use_raw_macro,
        raw_fred_csv=args.raw_fred_csv,
        raw_vix_json=args.raw_vix_json,
        raw_dxy_csv=args.raw_dxy_csv,
        raw_news_dir=args.raw_news_dir
    )
    
    print(f"\nDry run result: {result}")