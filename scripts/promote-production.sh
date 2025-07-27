#!/bin/bash

# Production promotion script
set -e

RELEASE_NAME=${1:-conviction-ai-pipeline}
NAMESPACE=${2:-production}
RUN_DATE=${3:-$(date -d "yesterday" +%Y-%m-%d)}

echo "🚀 Promoting to production..."
echo "Release: $RELEASE_NAME"
echo "Namespace: $NAMESPACE"
echo "Date: $RUN_DATE"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy to production with full resources
helm upgrade --install $RELEASE_NAME charts/conviction-ai-pipeline \
  --namespace $NAMESPACE \
  --set runDate=$RUN_DATE \
  --set chaos.enabled=false \
  --set backfill.enabled=true \
  --set resources.limits.cpu=8 \
  --set resources.limits.memory=16Gi \
  --set resources.limits.nvidia.com/gpu=1 \
  --set persistence.data.size=200Gi \
  --set persistence.models.size=50Gi \
  --set persistence.metrics.size=10Gi \
  --set lineage.enabled=true \
  --set lineage.host="lineage.prod.conviction-ai.com" \
  --wait --timeout=10m

echo "✅ Production deployment completed"