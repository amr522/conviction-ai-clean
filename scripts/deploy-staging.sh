#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${1:-staging}
RELEASE_NAME=${2:-conviction-ai-pipeline}
DATE=${3:-$(date -v-1d +%Y-%m-%d)}

echo "🔧 Creating namespace $NAMESPACE (if missing)…"
kubectl get ns $NAMESPACE || kubectl create ns $NAMESPACE

echo "🚀 Deploying to $NAMESPACE with minimal chaos…"
helm upgrade --install $RELEASE_NAME charts/conviction-ai-pipeline \
  --namespace $NAMESPACE \
  --set chaos.enabled=true \
  --set chaos.etl.duration=10 \
  --set chaos.etl.percentAffected=1 \
  --set chaos.inference.duration=10 \
  --set chaos.inference.percentAffected=1 \
  --set runDate=$DATE \
  --set backfill.enabled=false

echo "📡 Waiting for pods to be ready…"
kubectl rollout status deployment/$RELEASE_NAME -n $NAMESPACE

echo "✅ Staging deployment complete. Monitor chaos experiments via LitmusChaos dashboard."