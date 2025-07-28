#!/usr/bin/env bash
set -e

# M2 Ultra Optimization Settings with GPU Acceleration
export PYARROW_MEMORY_POOL=jemalloc  # Optimize memory allocation for large datasets
export POLARS_MAX_THREADS=24         # Use all 24 cores for Polars operations
export OMP_NUM_THREADS=24            # OpenMP threads for numerical operations
export MKL_NUM_THREADS=24            # Intel MKL threads (if available)
export NUMBA_NUM_THREADS=24          # Numba JIT compilation threads
export N_JOBS=24                     # Parallel job count for ML operations

# GPU optimization flags
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Enable PyTorch MPS with fallback
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Use all available GPU memory

echo "🚀 M2 Ultra Standalone Pipeline - 24 cores, 64GB RAM, Apple Metal GPU acceleration"
echo "🎯 GPU Backend: Apple Metal Performance Shaders (MPS) enabled"

DATE=$(python scripts/dry_run_schema_validation.py 2>&1 \
  | awk '/validation successful for/ {print $NF}')

if [ -z "$DATE" ]; then
  echo "❌ Failed to extract DATE from schema validation"
  exit 1
fi

echo "✅ Schema validation passed for $DATE"

# Check GPU availability
echo "🔍 Checking GPU availability..."
python -c "
from src.gpu_utils import gpu_supported, optimize_for_apple_silicon
optimize_for_apple_silicon()
if gpu_supported():
    print('✅ Apple Metal GPU acceleration available')
    print('🚀 GPU-accelerated pipeline mode enabled')
else:
    print('⚠️  GPU not available, using optimized CPU (24 cores)')
"

echo "🔄 Calculating features in standalone mode with M2 Ultra + GPU optimization..."

# Start performance monitoring
echo "📊 Starting performance monitoring..."
START_TIME=$(date +%s)

python src/calculate_features.py \
  --date "$DATE" \
  --daily-master-path staged/daily_master.parquet \
  --intraday-master-path datasets/intraday_master.parquet \
  --output-path data/Parquet_data/features_${DATE}.parquet \
  --use-gpu \
  --n-jobs 24 \
  --window-days 30

# Calculate execution time
END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))
echo "⏱️  Feature calculation completed in ${EXECUTION_TIME} seconds"

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

echo "🎉 single_day_pipeline_standalone.sh complete for $DATE"
echo "📁 Generated files:"
echo "  - Features: $FEATURES_PATH"
echo "  - Labels: $LABELS_PATH"
echo "  - Training dataset: $TRAIN_DATASET_PATH"
