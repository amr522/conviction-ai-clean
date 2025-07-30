#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Validating autoscaling YAML templates..."

# Create temporary directory for isolated templates
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Copy only autoscaling-related templates
mkdir -p "$TEMP_DIR/templates"
cp charts/conviction-ai-pipeline/templates/_helpers.tpl "$TEMP_DIR/templates/"
cp charts/conviction-ai-pipeline/templates/hpa.yaml "$TEMP_DIR/templates/"
cp charts/conviction-ai-pipeline/templates/vpa.yaml "$TEMP_DIR/templates/"
cp charts/conviction-ai-pipeline/templates/grafana-dashboard-costs.yaml "$TEMP_DIR/templates/"
cp charts/conviction-ai-pipeline/templates/deployment.yaml "$TEMP_DIR/templates/"
cp charts/conviction-ai-pipeline/templates/service.yaml "$TEMP_DIR/templates/"
cp charts/conviction-ai-pipeline/Chart.yaml "$TEMP_DIR/"
cp charts/conviction-ai-pipeline/values.yaml "$TEMP_DIR/"

# Test HPA template YAML generation
echo "Testing HPA template YAML generation..."
HPA_YAML=$(helm template conviction-ai-pipeline "$TEMP_DIR" \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=1 \
  --set autoscaling.maxReplicas=3 \
  --set autoscaling.cpu.targetAverageUtilization=50 \
  --show-only templates/hpa.yaml)

if echo "$HPA_YAML" | grep -q "kind: HorizontalPodAutoscaler"; then
    echo "✅ HPA template generates valid YAML"
else
    echo "❌ HPA template validation failed"
    exit 1
fi

# Test VPA template YAML generation
echo "Testing VPA template YAML generation..."
VPA_YAML=$(helm template conviction-ai-pipeline "$TEMP_DIR" \
  --set autoscaling.vpa.enabled=true \
  --set autoscaling.vpa.updateMode=Auto \
  --show-only templates/vpa.yaml)

if echo "$VPA_YAML" | grep -q "kind: VerticalPodAutoscaler"; then
    echo "✅ VPA template generates valid YAML"
else
    echo "❌ VPA template validation failed"
    exit 1
fi

# Test cost dashboard YAML generation
echo "Testing cost dashboard YAML generation..."
DASHBOARD_YAML=$(helm template conviction-ai-pipeline "$TEMP_DIR" \
  --set grafana.dashboard.costs.enabled=true \
  --show-only templates/grafana-dashboard-costs.yaml)

if echo "$DASHBOARD_YAML" | grep -q "kind: ConfigMap"; then
    echo "✅ Cost dashboard template generates valid YAML"
else
    echo "❌ Cost dashboard template validation failed"
    exit 1
fi

# Test combined configuration
echo "Testing combined autoscaling configuration..."
COMBINED_YAML=$(helm template conviction-ai-pipeline "$TEMP_DIR" \
  --set autoscaling.enabled=true \
  --set autoscaling.vpa.enabled=true \
  --set grafana.dashboard.costs.enabled=true)

if echo "$COMBINED_YAML" | grep -q "kind: HorizontalPodAutoscaler" && \
   echo "$COMBINED_YAML" | grep -q "kind: VerticalPodAutoscaler" && \
   echo "$COMBINED_YAML" | grep -q "kind: ConfigMap"; then
    echo "✅ Combined configuration generates valid YAML"
else
    echo "❌ Combined configuration validation failed"
    exit 1
fi

echo "🎉 All autoscaling YAML templates validated successfully!"
