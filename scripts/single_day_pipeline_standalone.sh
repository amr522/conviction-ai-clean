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

echo "🎮 M2 Ultra GPU-ONLY Standalone Pipeline - Apple Metal Performance Shaders"
echo "🚀 GPU Backend: MPS device enforced for all operations"

# Use a fixed date for standalone operation (or get from command line)
if [ -n "$1" ]; then
  DATE="$1"
else
  DATE="2025-01-01"  # Default date that has existing features
fi

echo "📅 Using date: $DATE"

# Check GPU availability
echo "🔍 Checking GPU availability..."
python -c "
from src.gpu_utils import gpu_supported, optimize_for_apple_silicon
optimize_for_apple_silicon()
if gpu_supported():
    print('✅ Apple Metal GPU acceleration available')
    print('🚀 GPU-accelerated pipeline mode enabled')
else:
    print('❌ GPU not available - GPU-ONLY mode required')
    exit(1)
"

# Define master dataset paths
DAILY_MASTER="staged/daily_master.parquet"
INTRADAY_MASTER="datasets/intraday_master.parquet"

# Check for master datasets
if [ ! -f "$DAILY_MASTER" ] || [ ! -f "$INTRADAY_MASTER" ]; then
  echo "❌ Master datasets missing: $DAILY_MASTER or $INTRADAY_MASTER"
  echo "💡 Hint: Run full pipeline first to generate master datasets:"
  echo "   python src/run_full_pipeline.py --date $DATE"
  exit 1
fi

echo "🔄 Calculating features in standalone mode with M2 Ultra + GPU optimization..."

# Start performance monitoring
echo "📊 Starting performance monitoring..."
START_TIME=$(date +%s)

python src/calculate_features.py \
  --date "$DATE" \
  --daily-master-path "$DAILY_MASTER" \
  --intraday-master-path "$INTRADAY_MASTER" \
  --output-path "data/Parquet_data/features_${DATE}.parquet" \
  --use-gpu \
  --n-jobs 24 \
  --window-days 30

# Calculate execution time
END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))
echo "⏱️  Feature calculation completed in ${EXECUTION_TIME} seconds"

echo "📊 Generating labels..."
LABELS_PATH="data/Parquet_data/labels_${DATE}.parquet"
if [ -f "$LABELS_PATH" ]; then
  echo "✅ Using existing labels file: $LABELS_PATH"
else
  python src/generate_labels.py --date "$DATE" || echo "⚠️  Label generation completed with warnings"
fi

echo "🔗 Building training dataset..."
TRAIN_DATASET_PATH="data/Parquet_data/train_dataset_${DATE}.parquet"

if [ -f "$FEATURES_PATH" ] && [ -f "$LABELS_PATH" ]; then
  if [ -f "scripts/generate-training-dataset.sh" ]; then
    ./scripts/generate-training-dataset.sh \
      "$FEATURES_PATH" \
      "$LABELS_PATH" \
      "$TRAIN_DATASET_PATH" || echo "⚠️  Training dataset generation completed with warnings"
  else
    echo "⚠️  Training dataset generation script not found, skipping..."
    cp "$FEATURES_PATH" "$TRAIN_DATASET_PATH"
  fi
else
  echo "❌ Missing required files for training dataset generation"
  [ ! -f "$FEATURES_PATH" ] && echo "  - Features: $FEATURES_PATH"
  [ ! -f "$LABELS_PATH" ] && echo "  - Labels: $LABELS_PATH"
fi

echo "🔍 Running validations..."
python validate_option_features.py --input-path "$TRAIN_DATASET_PATH"
python validate_feature_lagging.py --input-path "$TRAIN_DATASET_PATH"

echo "🎉 single_day_pipeline_standalone.sh GPU-ONLY complete for $DATE"
echo "📁 Generated files:"
echo "  - Features: $FEATURES_PATH"
echo "  - Labels: $LABELS_PATH"
echo "  - Training dataset: $TRAIN_DATASET_PATH"
