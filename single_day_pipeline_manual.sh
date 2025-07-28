#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting single-day pipeline (manual paths)..."

# 1. Run schema validation and capture DATE
DATE=$(
  python scripts/dry_run_schema_validation.py 2>&1 \
  | awk '/validation successful for/ { print $NF }'
)
echo "✅ Using DATE=$DATE"

# 2. Build features, labels, and train-dataset with explicit paths
echo "🔄 Building features with explicit paths..."
python src/calculate_features.py \
  --daily-master-path staged/daily_master.parquet \
  --intraday-master-path datasets/intraday_master.parquet \
  --output-path "data/Parquet_data/features_${DATE}.parquet" \
  --date "$DATE" \
  --use-gpu

echo "📊 Generating labels..."
python src/generate_labels.py --date "$DATE"

echo "🔗 Building training dataset..."
./scripts/generate-training-dataset.sh \
  "data/Parquet_data/features_${DATE}.parquet" \
  "data/Parquet_data/labels_${DATE}.parquet" \
  "data/Parquet_data/train_dataset_${DATE}.parquet"

# 3. Validate options and lagging
echo "🔍 Running validations..."
python validate_option_features.py --input-path "data/Parquet_data/train_dataset_${DATE}.parquet"
python validate_feature_lagging.py --input-path "data/Parquet_data/train_dataset_${DATE}.parquet"

echo "🎉 Single-day pipeline complete for $DATE"
