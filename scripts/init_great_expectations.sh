#!/usr/bin/env bash
set -euo pipefail

# Initialize Great Expectations for the project
echo "🔍 Initializing Great Expectations for data quality validation"

# Check if Great Expectations is installed
if ! python -c "import great_expectations" 2>/dev/null; then
    echo "❌ Great Expectations not found. Installing..."
    pip install great_expectations>=0.15.0
fi

# Initialize Great Expectations context if not exists
if [ ! -d "great_expectations" ]; then
    echo "📋 Initializing Great Expectations context..."
    python -c "
import great_expectations as gx
context = gx.get_context(project_root_dir='.')
print('✅ Great Expectations context initialized')
"
else
    echo "✅ Great Expectations context already exists"
fi

# Create expectation suites for each dataset type
echo "📝 Creating expectation suites..."

python -c "
import great_expectations as gx
from src.validate_data_quality import create_expectation_suite

context = gx.get_context()

# Create suites for each dataset type
dataset_types = ['options_daily', 'options_30min', 'stocks_daily', 'stocks_30min']

for dataset_type in dataset_types:
    suite_name = f'{dataset_type}_suite'
    try:
        create_expectation_suite(context, suite_name, dataset_type)
        print(f'✅ Created expectation suite: {suite_name}')
    except Exception as e:
        print(f'⚠️ Error creating suite {suite_name}: {e}')

print('🎉 Great Expectations setup completed!')
"

echo ""
echo "📋 Next steps:"
echo "1. Run data quality validation:"
echo "   python src/validate_data_quality.py --date 2025-01-16"
echo ""
echo "2. View validation reports in metrics/ directory"
echo ""
echo "3. Customize expectations by editing the suites in great_expectations/expectations/"
