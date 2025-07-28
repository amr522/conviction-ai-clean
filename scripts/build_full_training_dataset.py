#!/usr/bin/env python3
"""
Build Full Training Dataset - Conviction AI Pipeline
Creates partitioned training dataset from 2018 to present
Following ML_TRAINING_ROADMAP.md Day 1 requirements
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_command(cmd, check=True):
    """Run shell command with logging"""
    logger.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        logger.error(f"Command failed: {cmd}")
        logger.error(f"Error: {result.stderr}")
        sys.exit(1)
    return result


def process_date(date_str, output_dir):
    """Process a single date through the full pipeline"""
    logger.info(f"Processing date: {date_str}")

    # Step 1: Clean all data sources
    commands = [
        f"python src/clean_stocks_30min.py --date {date_str}",
        f"python src/clean_stocks_daily.py --date {date_str}",
        f"python src/clean_options_30min.py --date {date_str}",
        f"python src/clean_options_daily.py --date {date_str}",
        f"python src/clean_macro_data.py --date {date_str}",
        f"python src/clean_news.py --date {date_str}",
    ]

    for cmd in commands:
        try:
            run_command(cmd, check=False)  # Don't fail if some data missing
        except Exception as e:
            logger.warning(f"Command failed (continuing): {cmd} - {e}")

    # Step 2: Calculate features
    try:
        run_command(f"python src/calculate_features.py --date {date_str} --use-gpu")
    except Exception as e:
        logger.error(f"Feature calculation failed for {date_str}: {e}")
        return False

    # Step 3: Generate labels
    try:
        run_command(f"python src/generate_labels.py --date {date_str}")
    except Exception as e:
        logger.warning(f"Label generation failed for {date_str}: {e}")
        return False

    # Step 4: Create training dataset
    features_path = f"data/Parquet_data/features_{date_str}.parquet"
    labels_path = f"data/Parquet_data/labels_{date_str}.parquet"

    if os.path.exists(features_path) and os.path.exists(labels_path):
        # Create partitioned output directory
        partition_dir = f"{output_dir}/date={date_str}"
        os.makedirs(partition_dir, exist_ok=True)

        output_path = f"{partition_dir}/train_dataset_{date_str}.parquet"

        try:
            run_command(
                f"./scripts/generate-training-dataset.sh {features_path} {labels_path} {output_path}"
            )
            logger.info(f"✅ Successfully created training dataset for {date_str}")
            return True
        except Exception as e:
            logger.error(f"Training dataset creation failed for {date_str}: {e}")
            return False
    else:
        logger.warning(f"Missing features or labels for {date_str}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build full training dataset")
    parser.add_argument(
        "--start-date", default="2018-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/Parquet_data/training",
        help="Output directory for partitioned dataset",
    )
    parser.add_argument("--skip-weekends", action="store_true", help="Skip weekends")
    parser.add_argument("--resume-from", help="Resume from specific date (YYYY-MM-DD)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate date range
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    if args.resume_from:
        start_date = datetime.strptime(args.resume_from, "%Y-%m-%d")
        logger.info(f"Resuming from {args.resume_from}")

    current_date = start_date
    success_count = 0
    total_count = 0

    logger.info(
        f"Building training dataset from {start_date.date()} to {end_date.date()}"
    )
    logger.info(f"Output directory: {args.output_dir}")

    while current_date <= end_date:
        # Skip weekends if requested
        if args.skip_weekends and current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        date_str = current_date.strftime("%Y-%m-%d")
        total_count += 1

        # Check if already processed
        partition_dir = f"{args.output_dir}/date={date_str}"
        if os.path.exists(partition_dir):
            logger.info(f"⏭️  Skipping {date_str} (already exists)")
            success_count += 1
        else:
            if process_date(date_str, args.output_dir):
                success_count += 1

        current_date += timedelta(days=1)

    logger.info(
        f"🎉 Completed! Successfully processed {success_count}/{total_count} dates"
    )
    logger.info(f"Training dataset available at: {args.output_dir}")

    # Validation step
    if success_count > 0:
        logger.info("Running validation on sample partition...")
        sample_partition = f"{args.output_dir}/date={end_date.strftime('%Y-%m-%d')}"
        if os.path.exists(sample_partition):
            sample_file = list(Path(sample_partition).glob("*.parquet"))[0]
            run_command(
                f"python validate_option_features.py --input-path {sample_file}",
                check=False,
            )


if __name__ == "__main__":
    main()
