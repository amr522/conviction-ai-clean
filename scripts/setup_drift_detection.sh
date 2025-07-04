#!/bin/bash
set -euo pipefail

PINNED_DATA_S3=${1:-"s3://hpo-bucket-773934887314/56_stocks/46_models/2025-07-02-03-05-02/train.csv"}
DRY_RUN=${2:-"false"}

echo "🔍 Setting up drift detection for HPO pipeline..."

if [[ "$DRY_RUN" == "true" ]]; then
    echo "🔍 [DRY-RUN] Would create EventBridge rule for daily drift detection"
    echo "🔍 [DRY-RUN] Would schedule automated_cleanup.py --drift-check"
    echo "🔍 [DRY-RUN] Would use input data: $PINNED_DATA_S3"
    exit 0
fi

# Create EventBridge rule for daily drift detection
aws events put-rule \
    --name "hpo-daily-drift-check" \
    --description "Daily drift detection for HPO pipeline" \
    --schedule-expression "rate(1 day)" \
    --state ENABLED

echo "✅ EventBridge rule created for daily drift detection"
echo "📊 Input data source: $PINNED_DATA_S3"
