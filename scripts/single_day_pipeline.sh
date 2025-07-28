#!/usr/bin/env bash
set -e

# 1️⃣ Ensure MPS GPU is enabled - GPU-ONLY M2 Ultra execution
export PYTORCH_ENABLE_MPS=1
export POLARS_USE_GPU=1

# GPU availability check
if ! python - <<EOF
import torch
assert torch.backends.mps.is_available(), "MPS GPU not available"
print("✅ Apple Metal GPU (MPS) confirmed available")
EOF
then
  echo "❌ MPS GPU not available—abort!"
  exit 1
fi

echo "🎮 M2 Ultra GPU-ONLY Pipeline - Apple Metal Performance Shaders"
echo "🚀 GPU Backend: MPS device enforced for all operations"

# 1. Find validated DATE from dry-run validator
DATE=$(python scripts/dry_run_schema_validation.py 2>&1 \
  | awk '/validation successful for/ {print $NF}')

if [ -z "$DATE" ]; then
  echo "❌ Failed to extract DATE from schema validation"
  exit 1
fi

echo "✅ Schema validation passed for $DATE"

# 2. Run full pipeline
echo "🧪 Running dry-run validation on GPU..."
python src/run_full_pipeline.py --date "$DATE" --dry-run --device mps

echo "🔄 Running full pipeline with GPU-ONLY M2 Ultra execution..."
python src/run_full_pipeline.py --date "$DATE" \
  --raw-fred-csv "data/Parquet_data/Raw/FRED.csv" \
  --raw-vix-json "data/Parquet_data/Raw/vix_data.json" \
  --raw-dxy-csv "data/Parquet_data/Raw/DXY.csv" \
  --raw-news-dir "data/Parquet_data/Raw/news" \
  --device mps

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
echo "🔍 Running validations on GPU..."
python validate_option_features.py --input-path "$TRAIN_DATASET_PATH" --device mps
python validate_feature_lagging.py --input-path "$TRAIN_DATASET_PATH" --device mps

echo "🎉 single_day_pipeline.sh GPU-ONLY complete for $DATE"
echo "📁 Generated files:"
echo "  - Features: $FEATURES_PATH"
echo "  - Labels: $LABELS_PATH"
echo "  - Training dataset: $TRAIN_DATASET_PATH"
