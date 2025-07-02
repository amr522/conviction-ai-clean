#!/bin/bash
set -euo pipefail

# Usage: 
# ./promote_model.sh <model-name> <version> <stage>
#
# Example:
# ./promote_model.sh xgboost-56-stocks 1 Production

MODEL_NAME=${1:-"xgboost-56-stocks"}
VERSION=${2:-"1"}
STAGE=${3:-"Production"}

echo "🚀 Promoting model $MODEL_NAME version $VERSION to stage $STAGE..."

if python register_model.py promote --model-name "$MODEL_NAME" --version "$VERSION" --stage "$STAGE"; then
  echo "✅ Successfully promoted model $MODEL_NAME version $VERSION to $STAGE"
else
  echo "❌ Failed to promote model"
  exit 1
fi
