#!/usr/bin/env bash
set -euo pipefail

echo "🧪 Testing model explainability workflow..."

# Install required dependencies
pip install shap scikit-learn

# Run the explainability test
bash scripts/run-explain-test.sh