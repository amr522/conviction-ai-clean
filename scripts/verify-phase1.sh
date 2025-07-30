#!/usr/bin/env bash
set -euo pipefail

# 1. End-to-End Smoke Run
DATE=$(date -v-1d +%Y-%m-%d)
echo "👉 Verifying full pipeline for $DATE"
./scripts/evaluate_pipeline.sh $DATE
echo "✅ Evaluate pipeline passed"

# 2. Signal-Quality Dashboard Check (via Prometheus API)
PROM_URL="http://localhost:9090/api/v1/query"
for metric in gamma_coverage flow_accuracy vol_spike_detection; do
  echo "Querying $metric from Prometheus..."
  resp=$(curl -s "${PROM_URL}?query=${metric}%7Bjob%3D'pipeline'%7D")
  echo "$metric response: $resp"
done
echo "✅ Dashboard metrics reachable"

# 3. Nightly Backfill & Drift Baseline Refresh Check
echo "Checking backfill tasks in Prefect logs..."
prefect deployment run-history historical-backfill --limit 5

# Run drift detection against new baseline
BASELINE="data/Parquet_data/features_baseline.parquet"
CURRENT="data/Parquet_data/features_${DATE}.parquet"
echo "Running drift check against new baseline..."
python src/check_data_drift.py \
  --reference $BASELINE \
  --current $CURRENT \
  --drift-enabled false
echo "✅ Phase 1 verification complete!"
