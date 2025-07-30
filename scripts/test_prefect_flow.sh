#!/usr/bin/env bash
set -euo pipefail

# Test Prefect Auto-Retrain Flow locally
TEST_DATE=${1:-$(date +%Y-%m-%d)}

echo "🧪 Testing Prefect Auto-Retrain Flow for $TEST_DATE"

# Check if Prefect is installed
if ! command -v prefect &> /dev/null; then
    echo "❌ Prefect not found. Installing..."
    pip install prefect>=2.0.0 prefect-shell>=0.1.0
fi

# Create test data directories
mkdir -p logs datasets/processed metrics

# Create mock reference data for drift testing
echo "📊 Creating mock reference data..."
python -c "
import pandas as pd
import numpy as np
np.random.seed(42)
df = pd.DataFrame({
    'feature_1': np.random.normal(0, 1, 100),
    'feature_2': np.random.normal(5, 2, 100),
    'target': np.random.normal(0.02, 0.01, 100)
})
df.to_parquet('datasets/processed/reference.parquet')
print('✅ Mock reference data created')
"

# Create mock current data (no drift)
echo "📊 Creating mock current data (no drift)..."
python -c "
import pandas as pd
import numpy as np
np.random.seed(42)  # Same seed = no drift
df = pd.DataFrame({
    'feature_1': np.random.normal(0, 1, 100),
    'feature_2': np.random.normal(5, 2, 100),
    'target': np.random.normal(0.02, 0.01, 100)
})
df.to_parquet('datasets/processed/$TEST_DATE.parquet')
print('✅ Mock current data created (no drift)')
" TEST_DATE="$TEST_DATE"

# Test the flow locally
echo "🚀 Running Prefect flow test..."
python src/flows/auto_retrain_flow.py --date "$TEST_DATE"

echo "✅ Prefect flow test completed!"
echo ""
echo "📋 To deploy for production:"
echo "   ./scripts/deploy_prefect_flow.sh"
