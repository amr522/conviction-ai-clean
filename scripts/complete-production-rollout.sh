#!/bin/bash

# Complete production rollout workflow
set -e

RELEASE_NAME=${1:-conviction-ai-pipeline}
NAMESPACE=${2:-production}
RUN_DATE=${3:-2025-07-27}

echo "🚀 Starting complete production rollout..."
echo "=================================="

# Step 1: Run production promotion
echo "Step 1: Promoting to production..."
./scripts/promote-production.sh $RELEASE_NAME $NAMESPACE $RUN_DATE

# Step 2: Verify deployment
echo "Step 2: Verifying deployment..."
./scripts/verify-production.sh $NAMESPACE $RELEASE_NAME

# Step 3: Run smoke tests
echo "Step 3: Running smoke tests..."
./scripts/production-smoke-test.sh $RUN_DATE

# Step 4: Monitor alerts & metrics
echo "Step 4: Checking monitoring..."
./scripts/monitor-production.sh $NAMESPACE

# Step 5: Celebrate!
echo "=================================="
echo "🎉 Production rollout complete and validated! 🚀"
echo ""
echo "Access points:"
echo "- Lineage Explorer: https://lineage.prod.conviction-ai.com"
echo "- Grafana: kubectl port-forward -n monitoring svc/grafana 3000:80"
echo "- Prometheus: kubectl port-forward -n monitoring svc/prometheus 9090:9090"
echo ""
echo "Next steps:"
echo "- Monitor Grafana dashboards for ETL/training metrics"
echo "- Ensure Prometheus alerts aren't firing"
echo "- Check lineage explorer functionality"
echo "=================================="