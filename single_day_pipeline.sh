#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting single-day pipeline..."

# 1. Run schema validation and capture DATE
echo "🔍 Running schema validation..."
DATE=$(
  python scripts/dry_run_schema_validation.py 2>&1 \
  | awk '/validation successful for/ { print $NF }'
)

if [ -z "$DATE" ]; then
  echo "❌ Failed to extract DATE from schema validation"
  exit 1
fi

echo "✅ Using DATE=$DATE"

# 2. Run full pipeline to generate master datasets and features
echo "🔄 Running full pipeline to generate master datasets..."
python src/run_full_pipeline.py --date "$DATE" \
  --raw-fred-csv "data/Parquet_data/Raw/FRED.csv" \
  --raw-vix-json "data/Parquet_data/Raw/vix_data.json" \
  --raw-dxy-csv "data/Parquet_data/Raw/DXY.csv" \
  --raw-news-dir "data/Parquet_data/Raw/news"

# 3. Generate labels
echo "📊 Generating labels..."
python src/generate_labels.py --date "$DATE"

# 4. Build training dataset (combine features and labels)
echo "🔗 Building training dataset..."
FEATURES_PATH="data/Parquet_data/features_${DATE}.parquet"
LABELS_PATH="data/Parquet_data/labels_${DATE}.parquet"
TRAIN_DATASET_PATH="data/Parquet_data/train_dataset_${DATE}.parquet"

if [ -f "$FEATURES_PATH" ] && [ -f "$LABELS_PATH" ]; then
  ./scripts/generate-training-dataset.sh \
    "$FEATURES_PATH" \
    "$LABELS_PATH" \
    "$TRAIN_DATASET_PATH"
  echo "✅ Training dataset created: $TRAIN_DATASET_PATH"
else
  echo "❌ Missing required files:"
  [ ! -f "$FEATURES_PATH" ] && echo "  - Features: $FEATURES_PATH"
  [ ! -f "$LABELS_PATH" ] && echo "  - Labels: $LABELS_PATH"
  exit 1
fi

# 5. Validate options and lagging
echo "🔍 Validating options features..."
python validate_option_features.py --input-path "$TRAIN_DATASET_PATH"

echo "🔍 Validating feature lagging..."
python validate_feature_lagging.py --input-path "$TRAIN_DATASET_PATH"

echo "🎉 Single-day pipeline complete for $DATE"
echo "📁 Generated files:"
echo "  - Features: $FEATURES_PATH"
echo "  - Labels: $LABELS_PATH"
echo "  - Training dataset: $TRAIN_DATASET_PATH"
