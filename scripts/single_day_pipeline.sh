#!/usr/bin/env bash
set -e

# 1. Find validated DATE from dry-run validator
DATE=$(python scripts/dry_run_schema_validation.py 2>&1 \
  | awk '/validation successful for/ {print $NF}')

if [ -z "$DATE" ]; then
  echo "❌ Failed to extract DATE from schema validation"
  exit 1
fi

echo "✅ Schema validation passed for $DATE"

# 2. Run full pipeline
echo "🧪 Running dry-run validation..."
python src/run_full_pipeline.py --date "$DATE" --dry-run

echo "🔄 Running full pipeline..."
python src/run_full_pipeline.py --date "$DATE" \
  --raw-fred-csv "data/Parquet_data/Raw/FRED.csv" \
  --raw-vix-json "data/Parquet_data/Raw/vix_data.json" \
  --raw-dxy-csv "data/Parquet_data/Raw/DXY.csv" \
  --raw-news-dir "data/Parquet_data/Raw/news"

# 3. Generate training dataset
echo "🔗 Building training dataset..."
FEATURES_PATH="data/Parquet_data/features_${DATE}.parquet"
LABELS_PATH="data/Parquet_data/labels_${DATE}.parquet"
TRAIN_DATASET_PATH="data/Parquet_data/train_dataset_${DATE}.parquet"

if [ -f "$FEATURES_PATH" ] && [ -f "$LABELS_PATH" ]; then
  ./scripts/generate-training-dataset.sh \
    "$FEATURES_PATH" \
    "$LABELS_PATH" \
    "$TRAIN_DATASET_PATH"
else
  echo "❌ Missing required files:"
  [ ! -f "$FEATURES_PATH" ] && echo "  - Features: $FEATURES_PATH"
  [ ! -f "$LABELS_PATH" ] && echo "  - Labels: $LABELS_PATH"
  exit 1
fi

# 4. Validate
echo "🔍 Running validations..."
python validate_option_features.py --input-path "$TRAIN_DATASET_PATH"
python validate_feature_lagging.py --input-path "$TRAIN_DATASET_PATH"

echo "🎉 single_day_pipeline.sh complete for $DATE"
echo "📁 Generated files:"
echo "  - Features: $FEATURES_PATH"
echo "  - Labels: $LABELS_PATH"
echo "  - Training dataset: $TRAIN_DATASET_PATH"
