import argparse
import glob
import os
from datetime import timedelta
from pathlib import Path

import polars as pl
from prefect import task
from prefect.tasks import task_input_hash


@task(
    name="validate_pipeline",
    description="Validate all outputs from the pipeline processing",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    retries=3,
    retry_delay_seconds=60,
)
def run(date: str, dry_run: bool = False) -> dict:
    """
    Validate all pipeline outputs for data quality and consistency.

    Args:
        date: The processing date (for logging/tracking)
        dry_run: If True, only simulate validation

    Returns:
        dict: Validation results and statistics
    """
    try:
        print(f"Starting pipeline validation for date: {date}")

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        staged_dir = os.path.join(project_root, "staged")
        partitioned_dir = os.path.join(project_root, "partitioned")

        validation_results = {}

        # Check staged intermediate files
        staged_files = {
            "stocks_daily": "stocks_daily_clean.parquet",
            "stocks_30min": "stocks_30min_clean.parquet",
            "options_daily": "options_daily_clean.parquet",
            "options_30min": "options_30min_clean.parquet",
            "daily_master": "daily_master.parquet",
            "intraday_master": "intraday_master.parquet",
        }

        print("\nValidating staged files...")
        for key, filename in staged_files.items():
            filepath = os.path.join(staged_dir, filename)
            result = validate_dataset(filepath, key)
            validation_results[key] = result

        # Check partitioned output files
        print("\nValidating partitioned files...")
        partitioned_paths = {
            "daily": os.path.join(partitioned_dir, "daily"),
            "intraday": os.path.join(partitioned_dir, "intraday"),
        }

        for key, dirpath in partitioned_paths.items():
            if os.path.exists(dirpath):
                files = glob.glob(os.path.join(dirpath, "*.parquet"))
                print(f"\nFound {len(files)} {key} partition files")

                total_rows = 0
                total_nulls = {}

                for f in sorted(files):
                    result = validate_dataset(f, f"{key}_{os.path.basename(f)}")
                    total_rows += result["row_count"]

                    # Aggregate null counts across partitions
                    for col, nulls in result["null_counts"].items():
                        total_nulls[col] = total_nulls.get(col, 0) + nulls

                validation_results[f"{key}_partitions"] = {
                    "total_files": len(files),
                    "total_rows": total_rows,
                    "null_counts": total_nulls,
                }
            else:
                print(f"\nWARNING: Partition directory not found: {dirpath}")
                validation_results[f"{key}_partitions"] = None

        # Validate relationships between files
        print("\nValidating dataset relationships...")

        # Compare daily master with its partitions
        if validation_results["daily_master"] and validation_results.get(
            "daily_partitions"
        ):
            master_rows = validation_results["daily_master"]["row_count"]
            partition_rows = validation_results["daily_partitions"]["total_rows"]

            if master_rows != partition_rows:
                print(
                    f"WARNING: Daily row count mismatch - Master: {master_rows}, Partitions: {partition_rows}"
                )

        # Compare intraday master with its partitions
        if validation_results["intraday_master"] and validation_results.get(
            "intraday_partitions"
        ):
            master_rows = validation_results["intraday_master"]["row_count"]
            partition_rows = validation_results["intraday_partitions"]["total_rows"]

            if master_rows != partition_rows:
                print(
                    f"WARNING: Intraday row count mismatch - Master: {master_rows}, Partitions: {partition_rows}"
                )

        return {
            "status": "success",
            "date": date,
            "validation_results": validation_results,
        }

    except Exception as e:
        print(f"Error validating pipeline: {str(e)}")
        raise


def validate_dataset(filepath: str, name: str) -> dict:
    """Helper function to validate a single dataset."""
    print(f"\nValidating {name}...")

    if not os.path.exists(filepath):
        print(f"WARNING: File not found: {filepath}")
        return {
            "status": "error",
            "error": "File not found",
            "row_count": 0,
            "column_count": 0,
            "null_counts": {},
        }

    try:
        # Load and validate data
        df = pl.scan_parquet(filepath, extra_columns="ignore")
        data = df.collect()

        row_count = data.shape[0]
        col_count = data.shape[1]

        print(f"Rows: {row_count}")
        print(f"Columns: {col_count}")

        # Calculate null counts
        null_counts = {}
        for col in data.columns:
            nulls = data[col].null_count()
            null_pct = (nulls / row_count) * 100 if row_count > 0 else 0
            null_counts[col] = nulls
            print(f"{col}: {nulls} nulls ({null_pct:.2f}%)")

        return {
            "status": "success",
            "row_count": row_count,
            "column_count": col_count,
            "null_counts": null_counts,
        }

    except Exception as e:
        print(f"Error validating {name}: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "row_count": 0,
            "column_count": 0,
            "null_counts": {},
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate pipeline outputs")
    parser.add_argument(
        "--date", type=str, required=True, help="Processing date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without writing files"
    )

    args = parser.parse_args()
    result = run(args.date, dry_run=args.dry_run)
    print(f"Task result: {result}")
