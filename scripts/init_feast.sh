#!/usr/bin/env bash
set -euo pipefail

# Initialize Feast feature store
echo "🍽️ Initializing Feast feature store"

# Check if Feast is installed
if ! python -c "import feast" 2>/dev/null; then
    echo "❌ Feast not found. Installing..."
    pip install feast>=0.21.0
fi

# Initialize Feast repository if not exists
if [ ! -f "feature_repo/feature_store.yaml" ]; then
    echo "📋 Initializing Feast repository..."
    cd feature_repo
    feast init
    cd ..
else
    echo "✅ Feast repository already exists"
fi

# Apply feature definitions
echo "📝 Applying feature definitions..."
cd feature_repo

# Apply feature views and entities
feast apply

echo "✅ Feast feature store initialized successfully!"

cd ..

echo ""
echo "📋 Next steps:"
echo "1. Materialize features:"
echo "   python src/feast_materialize.py --action materialize --start-date 2025-01-01 --end-date 2025-01-16"
echo ""
echo "2. List feature views:"
echo "   python src/feast_materialize.py --action list"
echo ""
echo "3. Get online features:"
echo "   python src/feast_materialize.py --action get-online --ticker AAPL --features stocks_30min:close stocks_30min:volume"
echo ""
echo "4. Start Feast UI (optional):"
echo "   feast ui"
