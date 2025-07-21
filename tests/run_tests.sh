#!/bin/bash

# Run pytest with specified options
echo "Running tests for AWS ML pipeline scripts..."
cd "$(dirname "$0")/.."
python -m pytest tests/ --maxfail=1 --disable-warnings -q

# Check the exit code
if [ $? -eq 0 ]; then
    echo "All tests passed successfully!"
    exit 0
else
    echo "Tests failed. Please check the output above."
    exit 1
fi
