#!/bin/bash
set -euo pipefail

# Validate Helm chart with rollout configuration
echo "🔍 Validating Helm chart with Argo Rollouts..."

CHART_DIR="charts/conviction-ai-pipeline"
TEMP_DIR=$(mktemp -d)

# Test 1: Validate chart syntax
echo "📋 Testing chart syntax..."
helm lint "$CHART_DIR"

# Test 2: Template with rollouts disabled (default)
echo "📋 Testing template with rollouts disabled..."
helm template test-release "$CHART_DIR" \
  --set rollout.enabled=false \
  --output-dir "$TEMP_DIR/disabled"

# Verify deployment exists when rollouts disabled
if ! grep -q "kind: Deployment" "$TEMP_DIR/disabled/conviction-ai-pipeline/templates/inference-deployment.yaml"; then
  echo "❌ Deployment not found when rollouts disabled"
  exit 1
fi

# Test 3: Template with rollouts enabled
echo "📋 Testing template with rollouts enabled..."
helm template test-release "$CHART_DIR" \
  --set rollout.enabled=true \
  --output-dir "$TEMP_DIR/enabled"

# Verify rollout exists when enabled
if ! grep -q "kind: Rollout" "$TEMP_DIR/enabled/conviction-ai-pipeline/templates/rollout.yaml"; then
  echo "❌ Rollout not found when rollouts enabled"
  exit 1
fi

# Verify services exist
if ! grep -q "inference-stable" "$TEMP_DIR/enabled/conviction-ai-pipeline/templates/inference-services.yaml"; then
  echo "❌ Stable service not found"
  exit 1
fi

if ! grep -q "inference-canary" "$TEMP_DIR/enabled/conviction-ai-pipeline/templates/inference-services.yaml"; then
  echo "❌ Canary service not found"
  exit 1
fi

# Test 4: Validate analysis template
echo "📋 Testing analysis template..."
helm template test-release "$CHART_DIR" \
  --set rollout.enabled=true \
  --set rollout.canary.analysis.enabled=true \
  --output-dir "$TEMP_DIR/analysis"

if ! grep -q "kind: AnalysisTemplate" "$TEMP_DIR/analysis/conviction-ai-pipeline/templates/analysis.yaml"; then
  echo "❌ AnalysisTemplate not found when analysis enabled"
  exit 1
fi

# Test 5: Validate metrics queries
echo "📋 Testing Prometheus metrics queries..."
if ! grep -q "prediction_latency_seconds" "$TEMP_DIR/analysis/conviction-ai-pipeline/templates/analysis.yaml"; then
  echo "❌ Latency metric not found in analysis template"
  exit 1
fi

if ! grep -q "predictions_total" "$TEMP_DIR/analysis/conviction-ai-pipeline/templates/analysis.yaml"; then
  echo "❌ Predictions metric not found in analysis template"
  exit 1
fi

# Test 6: Validate canary steps configuration
echo "📋 Testing canary steps configuration..."
helm template test-release "$CHART_DIR" \
  --set rollout.enabled=true \
  --set rollout.canary.steps[0].weight=20 \
  --set rollout.canary.steps[0].pause="30s" \
  --set rollout.canary.steps[1].weight=100 \
  --output-dir "$TEMP_DIR/custom-steps"

if ! grep -q "setWeight: 20" "$TEMP_DIR/custom-steps/conviction-ai-pipeline/templates/rollout.yaml"; then
  echo "❌ Custom canary weight not applied"
  exit 1
fi

# Cleanup
rm -rf "$TEMP_DIR"

echo "✅ All Helm chart validations passed!"
echo "🚀 Chart is ready for Argo Rollouts canary deployments"