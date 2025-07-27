#!/usr/bin/env bash
set -euo pipefail

FEATURES_PATH=${1:-}
LABELS_PATH=${2:-}
TRAIN_PATH=${3:-}

if [[ -z "$FEATURES_PATH" || -z "$LABELS_PATH" || -z "$TRAIN_PATH" ]]; then
    echo "Usage: $0 <features_path> <labels_path> <train_path>"
    exit 1
fi

echo "🔗 Generating training dataset..."
echo "Features: $FEATURES_PATH"
echo "Labels: $LABELS_PATH"
echo "Output: $TRAIN_PATH"

if [[ ! -f "$FEATURES_PATH" ]]; then
    echo "❌ Features file not found: $FEATURES_PATH"
    exit 1
fi

if [[ ! -f "$LABELS_PATH" ]]; then
    echo "❌ Labels file not found: $LABELS_PATH"
    exit 1
fi

python src/generate_training_dataset.py \
    --feature-path "$FEATURES_PATH" \
    --label-path "$LABELS_PATH" \
    --output-path "$TRAIN_PATH"

echo "✅ Training dataset generation completed"