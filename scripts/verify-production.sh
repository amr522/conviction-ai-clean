#!/bin/bash
set -euo pipefail

echo "🔍 Verifying production deployment..."

NAMESPACE="production"
APP_LABEL="app=conviction-ai-pipeline"

# Check pod status
echo "📊 Checking pod status..."
READY_PODS=$(kubectl get pods -n $NAMESPACE -l $APP_LABEL --no-headers | grep "Running" | wc -l)
TOTAL_PODS=$(kubectl get pods -n $NAMESPACE -l $APP_LABEL --no-headers | wc -l)

if [ "$READY_PODS" -eq 0 ] || [ "$READY_PODS" -ne "$TOTAL_PODS" ]; then
    echo "❌ Not all pods are running. Ready: $READY_PODS, Total: $TOTAL_PODS"
    kubectl get pods -n $NAMESPACE -l $APP_LABEL
    exit 1
fi

# Health check
echo "🏥 Performing health checks..."
SERVICE_IP=$(kubectl get svc conviction-ai-pipeline -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
if kubectl run health-check --rm -i --restart=Never --image=curlimages/curl -- \
    curl -f "http://$SERVICE_IP:8000/health" --max-time 10; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

# Check metrics endpoint
echo "📈 Checking metrics endpoint..."
if kubectl run metrics-check --rm -i --restart=Never --image=curlimages/curl -- \
    curl -f "http://$SERVICE_IP:8000/metrics" --max-time 10; then
    echo "✅ Metrics endpoint accessible"
else
    echo "❌ Metrics endpoint failed"
    exit 1
fi

echo "✅ Production verification completed successfully!"