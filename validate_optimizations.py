#!/usr/bin/env python3
"""
Validation script to test performance optimizations.
Run with profiling to verify speedups.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse

from src.run_full_pipeline import run_full_pipeline
from src.utils.profiling import (clear_profile_results, enable_profiling,
                                 save_profile_report)


def main():
    parser = argparse.ArgumentParser(description="Validate performance optimizations")
    parser.add_argument(
        "--date", type=str, required=True, help="Test date (YYYY-MM-DD)"
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling")

    args = parser.parse_args()

    if args.profile:
        enable_profiling()
        clear_profile_results()
        print("🔍 Profiling enabled for optimization validation")

    print(f"🚀 Testing optimized pipeline for {args.date}")

    # Run the optimized pipeline
    result = run_full_pipeline(
        date=args.date, dry_run=True, profile=args.profile  # Use dry run for testing
    )

    if result["status"] == "success":
        print("✅ Optimized pipeline completed successfully!")

        if args.profile:
            report_path = save_profile_report(f"{args.date}_optimized")
            print(f"📊 Optimization profile report: {report_path}")

            # Print key performance metrics
            from src.utils.profiling import PROFILE_RESULTS

            if PROFILE_RESULTS:
                print("\n🏆 Performance Summary:")
                total_time = sum(r["duration_seconds"] for r in PROFILE_RESULTS)
                print(f"  Total execution time: {total_time:.3f}s")

                # Show top 3 slowest functions
                sorted_results = sorted(
                    PROFILE_RESULTS, key=lambda x: x["duration_seconds"], reverse=True
                )
                print("  Top 3 slowest functions:")
                for i, result in enumerate(sorted_results[:3], 1):
                    print(
                        f"    {i}. {result['function']}: {result['duration_seconds']:.3f}s"
                    )
    else:
        print("❌ Pipeline failed:", result.get("error", "Unknown error"))
        sys.exit(1)


if __name__ == "__main__":
    main()
