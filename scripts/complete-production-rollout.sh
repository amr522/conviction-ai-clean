#!/bin/bash
set -euo pipefail

echo "🚀 Starting complete production rollout..."

# Check prerequisites
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl."
    exit 1
fi

if ! command -v helm &> /dev/null; then
    echo "❌ helm not found. Please install helm."
    exit 1
fi

# Deploy to production
# Verify kubectl can reach the cluster
if ! kubectl version --short >/dev/null 2>&1; then
  echo "❌ ERROR: Kubernetes cluster unreachable. Please set KUBECONFIG or ensure cluster is running." >&2
  exit 1
fi
echo "📦 Deploying to production namespace..."
helm upgrade --install conviction-ai-pipeline charts/conviction-ai-pipeline \
    --namespace production \
    --create-namespace \
    --set image.tag="${IMAGE_TAG:-latest}" \
    --set environment=production \
    --wait --timeout=10m

# Wait for rollout
echo "⏳ Waiting for rollout to complete..."
kubectl rollout status deployment/conviction-ai-pipeline -n production --timeout=600s

# Verify services
echo "🔍 Verifying services..."
kubectl get pods -n production -l app=conviction-ai-pipeline
kubectl get svc -n production

echo "✅ Production rollout completed successfully!"
