#!/usr/bin/env bash
set -euo pipefail

echo "🧪 Testing model explainability workflow..."

DATE=2025-01-01

# Create minimal test data
mkdir -p data/Parquet_data models

# Create test features
python -c "
import polars as pl
import numpy as np
from datetime import date

# Create test features
np.random.seed(42)
features = pl.DataFrame({
    'date': [date(2025, 1, 1)] * 100,
    'ticker': ['AAPL'] * 100,
    'f1': np.random.normal(0, 1, 100),
    'f2': np.random.normal(5, 2, 100),
    'f3': np.random.exponential(1, 100)
})
features.write_parquet('data/Parquet_data/features_${DATE}.parquet')
print('✅ Test features created')
"

# Create mock model
python -c "
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Create and train a simple model
X = np.random.random((100, 3))
y = np.random.random(100)
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)

# Save model
with open('models/latest.pkl', 'wb') as f:
    pickle.dump(model, f)
print('✅ Mock model created')
"

# Test SHAP explanations
echo "🔍 Testing SHAP explanations..."
python -c "
import sys, os
sys.path.insert(0, 'src')
from inference import explain_predictions, load_model
import polars as pl

try:
    model = load_model('models/latest.pkl')
    feats = pl.read_parquet('data/Parquet_data/features_${DATE}.parquet')
    shap_summary = explain_predictions(model, feats, None)
    
    assert isinstance(shap_summary, dict), 'SHAP summary should be dict'
    assert len(shap_summary) > 0, 'SHAP summary should not be empty'
    assert all(isinstance(v, (float, int)) for v in shap_summary.values()), 'All SHAP values should be numeric'
    
    print(f'✅ SHAP explanations computed for {len(shap_summary)} features')
    top_feature = max(shap_summary.items(), key=lambda x: x[1])
    print(f'Top feature: {top_feature[0]} = {top_feature[1]:.4f}')
    
except Exception as e:
    print(f'❌ SHAP test failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

# Test inference script
echo "🚀 Testing inference script..."
python src/inference.py \
    --model-path models/latest.pkl \
    --feature-path data/Parquet_data/features_${DATE}.parquet \
    --output-path test_predictions.parquet

if [[ -f "test_predictions.parquet" ]]; then
    echo "✅ Inference script completed successfully"
    
    # Verify predictions file
    python -c "
import polars as pl
df = pl.read_parquet('test_predictions.parquet')
assert 'prediction' in df.columns, 'Predictions column missing'
print(f'✅ Predictions file contains {df.shape[0]} rows with prediction column')
"
else
    echo "❌ Inference script failed to create output file"
    exit 1
fi

# Clean up test files
rm -f data/Parquet_data/features_${DATE}.parquet
rm -f models/latest.pkl
rm -f test_predictions.parquet

echo "🎉 Explainability smoke test passed!"