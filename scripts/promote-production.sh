#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-production}
RELEASE=${RELEASE:-conviction-ai-pipeline}

echo "🚀 Promoting to production: $RELEASE"

# Check if production namespace exists
if ! kubectl get namespace $NAMESPACE &>/dev/null; then
    echo "📦 Creating production namespace..."
    kubectl create namespace $NAMESPACE
fi

# Deploy to production
echo "🔄 Deploying to production..."
helm upgrade $RELEASE charts/conviction-ai-pipeline \
    --install \
    --namespace $NAMESPACE \
    --set image.tag=latest \
    --set autoscaling.enabled=true \
    --set drift.enabled=true \
    --wait

# Verify deployment
echo "✅ Verifying deployment..."
kubectl -n $NAMESPACE rollout status deployment/$RELEASE --timeout=300s

# Check health
echo "🏥 Health check..."
kubectl -n $NAMESPACE get pods -l app.kubernetes.io/name=conviction-ai-pipeline

echo "🎉 Production deployment complete!"
