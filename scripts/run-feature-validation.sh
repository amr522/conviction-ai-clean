#!/bin/bash
set -euo pipefail

# Cleanup on failure
trap 'echo "💥 Failure detected, cleaning up..."; rm -f data/Parquet_data/features_validation.parquet data/Parquet_data/daily_master.parquet data/Parquet_data/intraday_master.parquet; exit 1' ERR

echo "Running feature validation..."

# Generate sample masters
python scripts/generate_sample_masters.py

# Run feature calculation
python src/calculate_features.py \
  --daily-master-path data/Parquet_data/daily_master.parquet \
  --intraday-master-path data/Parquet_data/intraday_master.parquet \
  --output-path data/Parquet_data/features_validation.parquet \
  --date 2025-01-01 \
  --window-days 7 \
  --n-jobs 2

# Validate feature matrix
python src/validate_features.py \
  --features-list docs/features_list.md \
  --feature-table data/Parquet_data/features_validation.parquet

echo "✅ Feature validation completed"