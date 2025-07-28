#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 YYYY-MM-DD"
  echo "Example: $0 2025-07-27"
  exit 1
fi

DATE=$1
echo "✅ Running manual pipeline for $DATE"

echo "🔄 Calculating features with explicit paths..."
python src/calculate_features.py \
  --date "$DATE" \
  --daily-master-path staged/daily_master.parquet \
  --intraday-master-path datasets/intraday_master.parquet \
  --output-path data/Parquet_data/features_${DATE}.parquet \
  --use-gpu

echo "📊 Generating labels..."
python src/generate_labels.py --date "$DATE"

echo "🔗 Building training dataset..."
FEATURES_PATH="data/Parquet_data/features_${DATE}.parquet"
LABELS_PATH="data/Parquet_data/labels_${DATE}.parquet"
TRAIN_DATASET_PATH="data/Parquet_data/train_dataset_${DATE}.parquet"

./scripts/generate-training-dataset.sh \
  "$FEATURES_PATH" \
  "$LABELS_PATH" \
  "$TRAIN_DATASET_PATH"

echo "🔍 Running validations..."
python validate_option_features.py --input-path "$TRAIN_DATASET_PATH"
python validate_feature_lagging.py --input-path "$TRAIN_DATASET_PATH"

echo "🎉 single_day_pipeline_manual.sh complete for $DATE"
echo "📁 Generated files:"
echo "  - Features: $FEATURES_PATH"
echo "  - Labels: $LABELS_PATH"
echo "  - Training dataset: $TRAIN_DATASET_PATH"
