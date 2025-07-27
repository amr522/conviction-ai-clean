#!/bin/bash

# Production verification script
set -e

NAMESPACE=${1:-production}
RELEASE_NAME=${2:-conviction-ai-pipeline}

echo "🔍 Verifying production deployment..."

# Check deployment status
echo "Checking rollout status..."
kubectl -n $NAMESPACE rollout status deployment/$RELEASE_NAME --timeout=300s

# Check pods
echo "Checking pod status..."
kubectl -n $NAMESPACE get pods -l app.kubernetes.io/name=conviction-ai-pipeline

# Check services
echo "Checking services..."
kubectl -n $NAMESPACE get svc -l app.kubernetes.io/name=conviction-ai-pipeline

# Check ingress
echo "Checking ingress..."
kubectl -n $NAMESPACE get ingress -l app.kubernetes.io/name=conviction-ai-pipeline

# Check persistent volumes
echo "Checking persistent volumes..."
kubectl -n $NAMESPACE get pvc -l app.kubernetes.io/name=conviction-ai-pipeline

echo "✅ Production verification completed"