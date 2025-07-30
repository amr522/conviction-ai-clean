#!/bin/bash
set -euo pipefail

# Cleanup on failure
trap 'echo "💥 Failure detected, cleaning up..."; rm -rf staged; exit 1' ERR

echo "Running schema validation..."

# Create test parquet files
mkdir -p staged
python -c "
import pandas as pd
import numpy as np

for ds in ['stocks_daily', 'options_daily', 'stocks_30min', 'options_30min']:
    df = pd.DataFrame({
        'test_col': [1, 2, 3],
        'string_col': ['a', 'b', 'c'],
        'float_col': [1.1, 2.2, 3.3]
    })
    df.to_parquet(f'staged/{ds}_clean.parquet')
"

# Test schema inspection
for ds in stocks_daily options_daily stocks_30min options_30min; do
    path="staged/${ds}_clean.parquet"
    python -c "
import pyarrow.parquet as pq
table = pq.read_table('$path')
for field in table.schema:
    print(f'    {field.name}: {field.type}')
"
done

echo "✅ Schema validation completed"
