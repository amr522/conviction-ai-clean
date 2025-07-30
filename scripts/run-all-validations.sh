#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Running schema validation..."
./scripts/run-schema-validation.sh

echo "✨ Running feature smoke tests..."
./scripts/run-feature-smoke-test.sh

echo "🔢 Running feature list validation..."
./scripts/run-feature-validation.sh

echo "⚙️  Running calculate features tests..."
./scripts/run-calculate-features-test.sh

echo "🤖 Running training CLI tests..."
./scripts/run-train-cli-test.sh

echo "🚀 Running performance utils tests..."
./scripts/run-performance-utils-test.sh

echo "🔧 Running performance utils extra tests..."
pytest tests/test_performance_utils_extra.py -v

echo "🔬 Running advanced signal validation..."
# Create test data if it doesn't exist
mkdir -p data/Parquet_data
if [[ ! -f "data/Parquet_data/options_30min_clean_$(date +%F).parquet" ]]; then
    python -c "
import polars as pl
from datetime import date

df = pl.DataFrame({
    'ticker': ['AAPL'] * 100,
    'timestamp': list(range(100)),
    'opt30_net_gamma': [1.0] * 90 + [None] * 10,
    'opt30_flow_divergence': [1.0] * 50 + [-1.0] * 50,
    'opt30_mid_price': list(range(100, 200)),
    'opt30_volume': [100] * 100,
    'opt30_vol_mean_5': [50] * 100,
    'opt30_vol_std_5': [10] * 100,
    'opt30_vol_spike': [True] * 20 + [False] * 80
})
df.write_parquet('data/Parquet_data/options_30min_clean_$(date +%F).parquet')
"
fi

./src/validate_advanced_signals.py \
  --input data/Parquet_data/options_30min_clean_$(date +%F).parquet \
  --threshold 0.8

echo "✅ All local validations passed!"
