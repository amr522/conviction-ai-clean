#!/bin/bash
set -euo pipefail

# Telegram notification helper
send_telegram_alert() {
    local status="$1"
    local payload="$2"
    python -c "from src.telegram_alerts import send_message; send_message('$status','$payload')" 2>/dev/null || echo "⚠️ Telegram alert failed"
}

RUN_DATE=${1:-2025-07-27}
NAMESPACE="production"

echo "🧪 Running production smoke tests for $RUN_DATE..."

# Test core pipeline functionality
echo "📊 Testing pipeline evaluation..."
./scripts/evaluate_pipeline.sh $RUN_DATE

# Test API endpoints
echo "🔌 Testing API endpoints..."
SERVICE_IP=$(kubectl get svc conviction-ai-pipeline -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "localhost")

# Health check
curl -f "http://$SERVICE_IP:8000/health" --max-time 10 || {
    echo "❌ Health endpoint failed"
    exit 1
}

# Metrics check
curl -f "http://$SERVICE_IP:8000/metrics" --max-time 10 || {
    echo "❌ Metrics endpoint failed"
    exit 1
}

# Test lineage explorer if deployed
echo "🔍 Testing lineage explorer..."
if kubectl -n $NAMESPACE get ingress conviction-ai-pipeline-lineage >/dev/null 2>&1; then
    echo "✅ Lineage explorer deployed"
    # Test lineage health
    LINEAGE_IP=$(kubectl get svc conviction-ai-pipeline-lineage -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")
    if [ -n "$LINEAGE_IP" ]; then
        curl -f "http://$LINEAGE_IP:8000/health" --max-time 10 && echo "✅ Lineage health OK" || echo "⚠️ Lineage health check failed"
    fi
else
    echo "⚠️ Lineage explorer not deployed"
fi

# Validate signal metrics
# Disable AWS XRay if metadata unavailable
export AWS_XRAY_DAEMON_ADDRESS=${AWS_XRAY_DAEMON_ADDRESS:-none}
export AWS_XRAY_TRACING_NAME=${AWS_XRAY_TRACING_NAME:-none}
export AWS_XRAY_SDK_ENABLED=false

echo "📈 Validating signal metrics..."
python src/validate_signals.py --threshold 0.8 || {
    echo "❌ Signal validation failed"
    send_telegram_alert "SIGNAL VALIDATION FAILED" "Signal validation failed in production smoke test"
    exit 1
}

echo "✅ Production smoke tests completed successfully!"
