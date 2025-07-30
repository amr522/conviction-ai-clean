#!/usr/bin/env bash
set -euo pipefail

DATE=2025-07-27

echo "🧪 Testing training dataset generation for $DATE"

# Create mock labels file for testing
mkdir -p data/Parquet_data
python -c "
import polars as pl
from datetime import date

# Create mock labels
labels = pl.DataFrame({
    'date': [date(2025, 7, 27), date(2025, 7, 27)],
    'ticker': ['AAPL', 'MSFT'],
    'target': [0.02, -0.01],
    'iv_change_5d': [0.05, -0.03]
})
labels.write_parquet('data/Parquet_data/labels_${DATE}.parquet')
print('✅ Mock labels created')
"

# Run full pipeline with schema check (dry run to avoid external dependencies)
python src/run_full_pipeline.py --date $DATE --check-schema --dry-run

# Test standalone training dataset generation
if [[ -f "data/Parquet_data/features_${DATE}.parquet" && -f "data/Parquet_data/labels_${DATE}.parquet" ]]; then
    echo "📊 Testing standalone training dataset generation..."

    python src/generate_training_dataset.py \
        --feature-path "data/Parquet_data/features_${DATE}.parquet" \
        --label-path "data/Parquet_data/labels_${DATE}.parquet" \
        --output-path "data/Parquet_data/train_dataset_${DATE}.parquet"

    # Verify training dataset file exists
    if [[ -f "data/Parquet_data/train_dataset_${DATE}.parquet" ]]; then
        echo "✅ Training dataset file generated successfully"

        # Verify dataset structure
        python -c "
import polars as pl
df = pl.read_parquet('data/Parquet_data/train_dataset_${DATE}.parquet')
print(f'Training dataset shape: {df.shape}')
if 'target' in df.columns:
    print('✅ Target column present')
else:
    print('❌ Target column missing')
    exit(1)
"
    else
        echo "❌ Training dataset file not found"
        exit 1
    fi
else
    echo "⚠️  Required files not found, skipping standalone test"
fi

# Clean up test files
rm -f data/Parquet_data/labels_${DATE}.parquet
rm -f data/Parquet_data/features_${DATE}.parquet
rm -f data/Parquet_data/train_dataset_${DATE}.parquet

echo "🎉 Training dataset test completed"
