#!/bin/bash
set -euo pipefail

# Usage: 
# ./list_registered_models.sh [model-name]
#
# Example:
# ./list_registered_models.sh
# ./list_registered_models.sh xgboost-56-stocks

MODEL_NAME=${1:-""}

echo "📋 Listing registered models..."

if [ -z "$MODEL_NAME" ]; then
  # List all models
  python register_model.py list
else
  # List versions of a specific model
  python register_model.py list --model-name "$MODEL_NAME"
fi
