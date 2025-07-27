#!/bin/bash
set -euo pipefail

# Cleanup on failure
trap 'echo "💥 Failure detected, cleaning up..."; rm -f data/Parquet_data/features_test.parquet data/Parquet_data/daily_master.parquet data/Parquet_data/intraday_master.parquet; exit 1' ERR

echo "Running feature smoke test..."

# Generate sample masters
python scripts/generate_sample_masters.py

# Run feature calculation
python src/calculate_features.py \
  --daily-master-path data/Parquet_data/daily_master.parquet \
  --intraday-master-path data/Parquet_data/intraday_master.parquet \
  --output-path data/Parquet_data/features_test.parquet \
  --date 2025-01-01 \
  --window-days 7 \
  --n-jobs 2

# Validate features file
python - <<<'import polars as pl; df = pl.read_parquet("data/Parquet_data/features_test.parquet"); assert "fred_rate_mean" in df.columns and df.height>0'

echo "✅ Feature smoke test completed"