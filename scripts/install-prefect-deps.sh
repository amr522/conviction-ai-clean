#!/usr/bin/env bash
set -euo pipefail

echo "Installing Prefect dependencies..."

# Check if prefect is installed
if ! python -c "import prefect" 2>/dev/null; then
    echo "Installing Prefect..."
    pip install prefect>=2.0.0
else
    echo "✅ Prefect already installed"
fi

# Verify installation
python -c "import prefect; print(f'Prefect version: {prefect.__version__}')"

echo "✅ Prefect dependencies ready"
