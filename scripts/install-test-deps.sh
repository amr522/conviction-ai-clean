#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Installing test dependencies..."
pip install -r requirements-dev.txt

echo "✅ Test dependencies installed successfully!"
echo "📊 Ready to run tests and coverage"