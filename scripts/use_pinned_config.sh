#!/bin/bash

set -euo pipefail

PINNED_DIR="models/pinned_successful_hpo"

if [[ ! -d "$PINNED_DIR" ]]; then
    echo "❌ Pinned configuration directory not found: $PINNED_DIR"
    exit 1
fi

echo "🔒 Using pinned successful HPO configuration:"
echo "📁 Directory: $PINNED_DIR"

DATASET_URI=$(jq -r '.dataset_uri' "$PINNED_DIR/hpo_config_pinned.json")
echo "📊 Dataset: $DATASET_URI"

export PINNED_DATASET_URI="$DATASET_URI"
echo "✅ PINNED_DATASET_URI exported: $PINNED_DATASET_URI"
