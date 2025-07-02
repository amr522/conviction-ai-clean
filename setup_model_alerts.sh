#!/bin/bash
set -euo pipefail

# Usage: 
# ./setup_model_alerts.sh <endpoint-name> <email>
#
# Example:
# ./setup_model_alerts.sh xgboost-56-stocks-endpoint alerts@example.com

ENDPOINT_NAME=${1:-""}
EMAIL=${2:-""}

if [ -z "$ENDPOINT_NAME" ] || [ -z "$EMAIL" ]; then
  echo "❌ Error: Missing required parameters"
  echo "Usage: ./setup_model_alerts.sh <endpoint-name> <email>"
  exit 1
fi

# Baseline URI is where the model's baseline statistics and constraints are stored
BASELINE_URI="s3://${S3_BUCKET:-hpo-bucket-773934887314}/baselines/${ENDPOINT_NAME}"

echo "🔔 Setting up monitoring and alerts for endpoint $ENDPOINT_NAME..."

if python setup_monitoring.py setup-all --endpoint-name "$ENDPOINT_NAME" --email "$EMAIL" --baseline-uri "$BASELINE_URI"; then
  echo "✅ Successfully set up monitoring and alerts for $ENDPOINT_NAME"
else
  echo "❌ Failed to set up monitoring and alerts"
  exit 1
fi
