#!/bin/bash
# Raw schema validation CI job

set -e

echo "🔍 Running raw schema validation tests..."

# Create test data and schema
mkdir -p test_data test_schemas

# Generate dummy parquet file
python3 -c "
import polars as pl
import json

# Create test data
test_data = pl.DataFrame({
    'ticker': ['AAPL220121C00150000', 'MSFT220121P00140000'],
    'close': [5.0, 3.0],
    'volume': [1000, 500],
    'timestamp': ['2025-01-15', '2025-01-15']
})

# Write test parquet
test_data.write_parquet('test_data/options_daily_2025-01-15.parquet')

# Create test schema
test_schema = {
    'type': 'object',
    'properties': {
        'ticker': {'type': 'string'},
        'close': {'type': 'number'},
        'volume': {'type': 'integer'},
        'timestamp': {'type': 'string'}
    },
    'required': ['ticker', 'close', 'volume']
}

# Write test schema
with open('test_schemas/options_daily_raw.json', 'w') as f:
    json.dump(test_schema, f, indent=2)

print('✅ Test data and schema created')
"

# Test the validator
python3 -c "
from src.utils.raw_schema_validator import validate, SchemaMismatchError

try:
    result = validate('test_data/options_daily_2025-01-15.parquet', 'test_schemas/options_daily_raw.json')
    print(f'✅ Schema validation test passed: {result}')
except Exception as e:
    print(f'❌ Schema validation test failed: {e}')
    exit(1)
"

# Run unit tests
python -m pytest tests/test_raw_schema_validator.py -v

# Cleanup
rm -rf test_data test_schemas

echo "✅ Raw schema validation tests completed successfully"