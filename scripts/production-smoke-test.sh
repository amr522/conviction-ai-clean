#!/bin/bash

# Production smoke test script
set -e

RUN_DATE=${1:-2025-07-27}

echo "🧪 Running production smoke tests..."
echo "Date: $RUN_DATE"

# Disable external services for smoke test
export S3_BUCKET=""
export SLACK_WEBHOOK_URL=""
export MLFLOW_TRACKING_URI=""
export OPENLINEAGE_URL=""

# Run pipeline evaluation
echo "Running pipeline evaluation..."
./scripts/evaluate_pipeline.sh $RUN_DATE

# Test lineage explorer health
echo "Testing lineage explorer..."
if kubectl -n production get ingress conviction-ai-pipeline-lineage >/dev/null 2>&1; then
    echo "✅ Lineage explorer ingress found"
else
    echo "⚠️ Lineage explorer not deployed"
fi

# Check metrics endpoints
echo "Checking metrics endpoints..."
kubectl -n production get servicemonitor -l component=lineage >/dev/null 2>&1 && echo "✅ Lineage metrics configured" || echo "⚠️ Lineage metrics not found"

echo "✅ Production smoke tests completed"
