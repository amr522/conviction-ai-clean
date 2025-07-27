#!/usr/bin/env bash
set -euo pipefail

# Raw macro data paths
RAW_FRED_CSV="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/FRED.csv"
RAW_VIX_JSON="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/vix_data.json"
RAW_DXY_CSV="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/DXY.csv"
RAW_NEWS_DIR="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/news"

# Environment defaults
export WINDOW_DAYS=${WINDOW_DAYS:-30}
export USE_GPU=${USE_GPU:-false}
export N_JOBS=${N_JOBS:-8}

START_DATE=${1:-$(date -v-7d +%Y-%m-%d)}
END_DATE=${2:-$(date -v-1d +%Y-%m-%d)}
MAX_WORKERS=${3:-24}

echo "🚀 Starting Prefect historical backfill..."
echo "  Start Date: $START_DATE"
echo "  End Date: $END_DATE"
echo "  Max Workers: $MAX_WORKERS"

# Ensure Prefect dependencies are installed
./scripts/install-prefect-deps.sh

# Run the Prefect flow with macro data paths
python src/flows/historical_backfill_flow.py \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --max-workers "$MAX_WORKERS" \
  --raw-fred-csv "$RAW_FRED_CSV" \
  --raw-vix-json "$RAW_VIX_JSON" \
  --raw-dxy-csv "$RAW_DXY_CSV" \
  --raw-news-dir "$RAW_NEWS_DIR"

echo "✅ Prefect backfill completed"