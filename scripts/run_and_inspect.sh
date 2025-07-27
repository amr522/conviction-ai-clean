#!/usr/bin/env bash
set -e

# Raw macro data paths
RAW_FRED_CSV="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/FRED.csv"
RAW_VIX_JSON="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/vix_data.json"
RAW_DXY_CSV="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/DXY.csv"
RAW_NEWS_DIR="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/news"

DATE=${1:-$(date -I -d "1 day ago")}  # default to yesterday's date
USE_RAW_MACRO=${USE_RAW_MACRO:-false}

echo "Running full pipeline for $DATE…"
if [[ "$USE_RAW_MACRO" == "true" ]]; then
    python src/run_full_pipeline.py --date $DATE --use-raw-macro \
        --raw-fred-csv "$RAW_FRED_CSV" \
        --raw-vix-json "$RAW_VIX_JSON" \
        --raw-dxy-csv "$RAW_DXY_CSV" \
        --raw-news-dir "$RAW_NEWS_DIR"
else
    python src/run_full_pipeline.py --date $DATE \
        --raw-fred-csv "$RAW_FRED_CSV" \
        --raw-vix-json "$RAW_VIX_JSON" \
        --raw-dxy-csv "$RAW_DXY_CSV" \
        --raw-news-dir "$RAW_NEWS_DIR"
fi
echo "Inspecting Parquet schemas:"
for ds in stocks_daily options_daily stocks_30min options_30min; do
  path="staged/${ds}_clean.parquet"
  if [[ ! -f "$path" ]]; then
    echo "⚠️  Warning: $path not found, skipping schema inspection for $ds" >&2
    continue
  fi
  
  echo "Schema for $ds:"
  python -c "
import pyarrow.parquet as pq
table = pq.read_table('$path')
for field in table.schema:
    print(f'    {field.name}: {field.type}')
"
  echo
done