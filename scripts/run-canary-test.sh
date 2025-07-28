#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-staging}
RELEASE=${RELEASE:-conviction-ai-pipeline}

echo "🚀 Starting canary rollout test for $RELEASE in $NAMESPACE"

# Check if rollout exists
if ! kubectl argo rollouts get rollout $RELEASE -n $NAMESPACE >/dev/null 2>&1; then
    echo "❌ Rollout $RELEASE not found in namespace $NAMESPACE"
    exit 1
fi

# Trigger canary deployment
echo "📦 Triggering canary deployment..."
kubectl -n $NAMESPACE set image rollout/$RELEASE $RELEASE=conviction-ai:canary-test

# Wait for canary to start
echo "⏳ Waiting for canary phase..."
kubectl -n $NAMESPACE argo rollouts get rollout $RELEASE --watch --timeout=300s | grep -q "Canary"

# Simulate failure by aborting rollout
echo "🔥 Simulating canary failure..."
kubectl -n $NAMESPACE argo rollouts abort $RELEASE

# Verify rollback
echo "🔄 Verifying rollback..."
kubectl -n $NAMESPACE argo rollouts get rollout $RELEASE --watch --timeout=300s | grep -q "Healthy"

echo "✅ Canary rollback test passed!"
