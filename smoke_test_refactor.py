#!/usr/bin/env python3
"""
Smoke test for refactored ETL scripts with performance utilities.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import tempfile
from datetime import datetime

import polars as pl

from src.run_full_pipeline import run_full_pipeline
from src.utils.profiling import (clear_profile_results, enable_profiling,
                                 save_profile_report)


def create_test_data():
    """Create minimal test data for smoke testing."""
    print("Creating test data...")

    # Create test directories
    os.makedirs("data/Parquet_data/option_minute", exist_ok=True)
    os.makedirs("data/Parquet_data/stocks_minute", exist_ok=True)
    os.makedirs("raw", exist_ok=True)

    # Sample timestamp
    ts = int(datetime(2025, 6, 1, 10, 0).timestamp() * 1e9)

    # Options minute data
    options_data = pl.DataFrame(
        {
            "window_start": [ts, ts],
            "ticker": ["AAPL250601C00150000", "AAPL250601P00150000"],
            "underlying": ["AAPL", "AAPL"],
            "open": [1.5, 0.8],
            "high": [1.6, 0.9],
            "low": [1.4, 0.7],
            "close": [1.55, 0.85],
            "volume": [100, 80],
            "transactions": [10, 8],
        }
    )
    options_data.write_parquet("data/Parquet_data/option_minute/test.parquet")

    # Stocks minute data
    stocks_data = pl.DataFrame(
        {
            "window_start": [ts, ts],
            "ticker": ["AAPL", "AAPL"],
            "open": [150.0, 150.0],
            "high": [151.0, 151.0],
            "low": [149.0, 149.0],
            "close": [150.5, 150.5],
            "volume": [10000, 10000],
            "transactions": [100, 100],
        }
    )
    stocks_data.write_parquet("data/Parquet_data/stocks_minute/test.parquet")

    # Daily options data
    daily_options = pl.DataFrame(
        {
            "window_start": [ts],
            "ticker": ["AAPL250601C00150000"],
            "underlying": ["AAPL"],
            "open": [1.5],
            "high": [1.6],
            "low": [1.4],
            "close": [1.55],
            "volume": [1000],
            "transactions": [100],
            "strike": [150.0],
            "option_type": ["C"],
        }
    )
    daily_options.write_parquet("raw/options_daily.parquet")

    print("✅ Test data created")


def main():
    print("🧪 Running smoke test for refactored ETL scripts")

    # Create test data
    create_test_data()

    # Enable profiling
    enable_profiling()
    clear_profile_results()

    # Run the refactored pipeline
    try:
        result = run_full_pipeline(date="2025-06-01", dry_run=True, profile=True)

        if result["status"] == "success":
            print("✅ Refactored pipeline completed successfully!")

            # Save profile report
            report_path = save_profile_report("refactor_smoke_test")
            print(f"📊 Profile report: {report_path}")

            # Verify performance utilities were used
            from src.utils.profiling import PROFILE_RESULTS

            function_names = [r["function"] for r in PROFILE_RESULTS]

            expected_functions = [
                "compute_flow_signals_optimized",
                "compute_gamma_signals_optimized",
                "optimize_join_performance",
            ]

            found_functions = [f for f in expected_functions if f in function_names]
            print(f"✅ Performance utilities used: {found_functions}")

            if len(found_functions) >= 2:
                print("🎉 Refactoring successful - performance utilities integrated!")
            else:
                print("⚠️  Some performance utilities not detected in profile")

        else:
            print("❌ Pipeline failed:", result.get("error", "Unknown error"))
            return 1

    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return 1

    # Cleanup test data
    import shutil

    for path in ["data", "raw", "staged", "logs"]:
        if os.path.exists(path):
            shutil.rmtree(path)

    print("🧹 Cleanup completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
