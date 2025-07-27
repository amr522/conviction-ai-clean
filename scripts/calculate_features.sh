#!/usr/bin/env bash
set -euo pipefail

# Feature calculation wrapper script
DATE=${1:-$(date +%Y-%m-%d)}
WINDOW_DAYS=${2:-30}
USE_GPU=${3:-false}
N_JOBS=${4:-$(nproc)}

echo "🧮 Calculating features for $DATE (window: $WINDOW_DAYS days, GPU: $USE_GPU, jobs: $N_JOBS)"

python src/calculate_features.py \
  --daily-master-path "staged/daily_master.parquet" \
  --intraday-master-path "datasets/intraday_master.parquet" \
  --output-path "datasets/features_${DATE}.parquet" \
  --date "$DATE" \
  --window-days "$WINDOW_DAYS" \
  --n-jobs "$N_JOBS" \
  $([ "$USE_GPU" = "true" ] && echo "--use-gpu" || echo "")

echo "✅ Features calculated: datasets/features_${DATE}.parquet"