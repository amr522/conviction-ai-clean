#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting single-day pipeline (standalone features)..."

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

# 2. Build features using standalone mode (requires master datasets to exist)
echo "🔄 Building features for $DATE..."
if python src/calculate_features.py --date "$DATE" --use-gpu; then
  echo "✅ Features generated successfully"
else
  echo "❌ Feature generation failed"
  echo "💡 Hint: You may need to run the full pipeline first:"
  echo "   python src/run_full_pipeline.py --date $DATE"
  exit 1
fi

# 3. Generate labels
echo "📊 Generating labels for $DATE..."
if python src/generate_labels.py --date "$DATE"; then
  echo "✅ Labels generated successfully"
else
  echo "❌ Label generation failed"
  exit 1
fi

# 4. Build training dataset
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
if python validate_option_features.py --input-path "$TRAIN_DATASET_PATH"; then
  echo "✅ Options validation passed"
else
  echo "⚠️ Options validation failed"
fi

echo "🔍 Validating feature lagging..."
if python validate_feature_lagging.py --input-path "$TRAIN_DATASET_PATH"; then
  echo "✅ Feature lagging validation passed"
else
  echo "⚠️ Feature lagging validation failed"
fi

echo "🎉 Single-day pipeline complete for $DATE"
echo "📁 Generated files:"
echo "  - Features: $FEATURES_PATH"
echo "  - Labels: $LABELS_PATH"
echo "  - Training dataset: $TRAIN_DATASET_PATH"
