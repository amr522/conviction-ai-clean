#!/usr/bin/env bash
set -e

# M2 Ultra Optimization Settings
export PYARROW_MEMORY_POOL=jemalloc  # Optimize memory allocation for large datasets
export POLARS_MAX_THREADS=24         # Use all 24 cores for Polars operations
export OMP_NUM_THREADS=24            # OpenMP threads for numerical operations
export MKL_NUM_THREADS=24            # Intel MKL threads (if available)
export NUMBA_NUM_THREADS=24          # Numba JIT compilation threads
export N_JOBS=24                     # Parallel job count for ML operations

echo "🚀 M2 Ultra Standalone Pipeline - 24 cores, 64GB RAM, GPU acceleration enabled"

DATE=$(python scripts/dry_run_schema_validation.py 2>&1 \
  | awk '/validation successful for/ {print $NF}')

if [ -z "$DATE" ]; then
  echo "❌ Failed to extract DATE from schema validation"
  exit 1
fi

echo "✅ Schema validation passed for $DATE"

echo "🔄 Calculating features in standalone mode with M2 Ultra optimization..."
python src/calculate_features.py \
  --date "$DATE" \
  --daily-master-path staged/daily_master.parquet \
  --intraday-master-path datasets/intraday_master.parquet \
  --output-path data/Parquet_data/features_${DATE}.parquet \
  --use-gpu \
  --n-jobs 24 \
  --window-days 30

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
