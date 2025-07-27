#!/usr/bin/env bash
set -euo pipefail

# Cleanup on failure
trap 'echo "💥 Failure detected, cleaning up..."; rm -f data/Parquet_data/features_test.parquet data/Parquet_data/daily_master.parquet data/Parquet_data/intraday_master.parquet; exit 1' ERR

DATE=${1:-$(date -v-1d +%Y-%m-%d)}
echo "🔍  Starting full evaluation for $DATE…"

echo "1️⃣ Dry‐run pipeline with schema checks"
python src/dry_run_pipeline.py --date $DATE --check-schema

echo "2️⃣ Inspect Parquet schemas"
./scripts/run_and_inspect.sh $DATE

echo "3️⃣ Generate sample masters for feature smoke‐test"
python scripts/generate_sample_masters.py

echo "4️⃣ Feature calculation smoke‐test"
python src/calculate_features.py \
  --daily-master-path data/Parquet_data/daily_master.parquet \
  --intraday-master-path data/Parquet_data/intraday_master.parquet \
  --output-path data/Parquet_data/features_test.parquet \
  --date $DATE --window-days 7 --n-jobs 2

python - <<<'import polars as pl; df=pl.read_parquet("data/Parquet_data/features_test.parquet"); \
  assert "fred_rate_mean" in df.columns and df.height>0; print("✅ Feature smoke‐test passed")'

echo "5️⃣ Quick train smoke‐test (5 trials, 4 workers)"
./scripts/run_and_train.sh $DATE 5 4

echo "🎉  All evaluation steps passed for $DATE"