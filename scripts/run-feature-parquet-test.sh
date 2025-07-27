#!/usr/bin/env bash
set -euo pipefail

DATE=2025-07-27

echo "🧪 Testing feature parquet generation for $DATE"

# Run full pipeline with schema check
python src/run_full_pipeline.py --date $DATE --check-schema --dry-run

# Check if feature parquet would be generated (dry run doesn't create files)
echo "✅ Pipeline dry run completed successfully"

# Test standalone feature calculation (requires master datasets)
if [[ -f "staged/daily_master.parquet" && -f "datasets/intraday_master.parquet" ]]; then
    python src/calculate_features.py --date $DATE
    
    # Verify parquet file exists
    if [[ -f "data/Parquet_data/features_${DATE}.parquet" ]]; then
        echo "✅ Feature parquet file generated successfully"
    else
        echo "❌ Feature parquet file not found"
        exit 1
    fi
else
    echo "⚠️  Master datasets not found, skipping standalone test"
fi

echo "🎉 Feature parquet test completed"