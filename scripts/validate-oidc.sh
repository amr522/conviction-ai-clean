#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Validating OIDC ingress templates..."

# Create temporary directory for isolated templates
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Copy required templates
mkdir -p "$TEMP_DIR/templates"
cp charts/conviction-ai-pipeline/templates/_helpers.tpl "$TEMP_DIR/templates/"
cp charts/conviction-ai-pipeline/templates/ingress-lineage.yaml "$TEMP_DIR/templates/"
cp charts/conviction-ai-pipeline/templates/oidc-secret.yaml "$TEMP_DIR/templates/"
cp charts/conviction-ai-pipeline/Chart.yaml "$TEMP_DIR/"
cp charts/conviction-ai-pipeline/values.yaml "$TEMP_DIR/"

# Test OIDC ingress template
echo "Testing OIDC ingress template..."
OIDC_YAML=$(helm template conviction-ai-pipeline "$TEMP_DIR" \
  --set lineage.enabled=true \
  --set lineage.auth.oidc.enabled=true \
  --set lineage.auth.oidc.clientSecret=test-secret \
  --show-only templates/ingress-lineage.yaml)

if echo "$OIDC_YAML" | grep -q "auth-url"; then
    echo "✅ OIDC ingress template generates valid YAML"
else
    echo "❌ OIDC ingress template validation failed"
    exit 1
fi

# Test OIDC secret template
echo "Testing OIDC secret template..."
SECRET_YAML=$(helm template conviction-ai-pipeline "$TEMP_DIR" \
  --set lineage.enabled=true \
  --set lineage.auth.oidc.enabled=true \
  --set lineage.auth.oidc.clientSecret=test-secret \
  --show-only templates/oidc-secret.yaml)

if echo "$SECRET_YAML" | grep -q "kind: Secret"; then
    echo "✅ OIDC secret template generates valid YAML"
else
    echo "❌ OIDC secret template validation failed"
    exit 1
fi

# Test basic auth fallback
echo "Testing basic auth fallback..."
BASIC_YAML=$(helm template conviction-ai-pipeline "$TEMP_DIR" \
  --set lineage.enabled=true \
  --set lineage.auth.oidc.enabled=false \
  --show-only templates/ingress-lineage.yaml)

if echo "$BASIC_YAML" | grep -q "auth-type.*basic"; then
    echo "✅ Basic auth fallback works correctly"
else
    echo "❌ Basic auth fallback validation failed"
    exit 1
fi

echo "🎉 OIDC templates validated successfully!"