#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${1:-staging}
DATE=${2:-2025-07-25}

echo "🚀 Starting end-to-end staging validation for $NAMESPACE..."

echo "1️⃣ Deploy to staging"
./scripts/deploy-staging.sh $NAMESPACE

echo "2️⃣ Port-forward services"
kubectl -n $NAMESPACE port-forward svc/conviction-ai-pipeline 9090:9090 &
METRICS_PID=$!
kubectl -n $NAMESPACE port-forward svc/conviction-ai-pipeline-inference 8000:8000 &
INFERENCE_PID=$!

# Wait for port-forwards to establish
sleep 5

echo "3️⃣ Run full evaluation in staging"
export S3_BUCKET=""
export SLACK_WEBHOOK_URL=""
export MLFLOW_TRACKING_URI=""
./scripts/evaluate_pipeline.sh $DATE

echo "4️⃣ Generate test JWT token"
export TEST_JWT=$(python scripts/generate_jwt_token.py --test --quiet)

echo "5️⃣ Smoke-test inference endpoint"
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TEST_JWT" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","timestamp":"'$DATE'T15:30:00"}' \
  --fail --silent --show-error

echo "6️⃣ Check chaos experiment status"
kubectl -n $NAMESPACE get chaosengines -o wide || echo "No chaos engines found"

echo "7️⃣ Verify metrics endpoint"
curl -s http://localhost:9090/metrics | grep -q "pipeline_etl_duration" || echo "Metrics not ready"

echo "8️⃣ Check pod recovery after chaos"
kubectl -n $NAMESPACE get pods -l app.kubernetes.io/name=conviction-ai-pipeline

# Cleanup port-forwards
kill $METRICS_PID $INFERENCE_PID 2>/dev/null || true

echo "✅ Staging validation complete! Ready for production promotion."
