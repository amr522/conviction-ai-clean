#!/usr/bin/env bash
set -euo pipefail

# Deploy Prefect Auto-Retrain Flow
FLOW_NAME="Auto-Retrain-Pipeline"
DEPLOYMENT_NAME="auto-retrain-daily"
CRON_SCHEDULE="0 2 * * *"  # Daily at 2 AM UTC

echo "🚀 Deploying Prefect Auto-Retrain Flow"

# Check if Prefect is installed
if ! command -v prefect &> /dev/null; then
    echo "❌ Prefect not found. Installing..."
    pip install prefect>=2.0.0
fi

echo "📋 Prefect version: $(prefect version)"

# Build deployment
echo "🔨 Building deployment..."
prefect deployment build src/flows/auto_retrain_flow.py:auto_retrain_flow \
    --name "$DEPLOYMENT_NAME" \
    --cron "$CRON_SCHEDULE" \
    --work-queue "default" \
    --params '{}' \
    --description "Daily auto-retrain pipeline with drift-based backfill" \
    --tags "auto-retrain,drift-detection,daily"

# Apply deployment
echo "📤 Applying deployment..."
prefect deployment apply auto_retrain_flow-deployment.yaml

echo "✅ Deployment created successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Start a Prefect agent:"
echo "   prefect agent start --work-queue default"
echo ""
echo "2. View deployment in UI:"
echo "   prefect server start"
echo "   # Then visit http://localhost:4200"
echo ""
echo "3. Trigger manual run:"
echo "   prefect deployment run '$FLOW_NAME/$DEPLOYMENT_NAME'"
echo ""
echo "4. Run with custom date:"
echo "   prefect deployment run '$FLOW_NAME/$DEPLOYMENT_NAME' --params '{\"target_date\": \"2025-01-16\"}'"