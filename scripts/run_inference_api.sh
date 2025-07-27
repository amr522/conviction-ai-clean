#!/usr/bin/env bash
set -euo pipefail

# Run FastAPI inference service locally
MODE=${1:-"dev"}
PORT=${2:-8000}
WORKERS=${3:-1}

echo "🚀 Starting Conviction AI Inference API"
echo "Mode: $MODE"
echo "Port: $PORT"
echo "Workers: $WORKERS"

# Set environment variables
export MODEL_PATH=${MODEL_PATH:-"models/latest.pkl"}
export FEAST_REPO_PATH=${FEAST_REPO_PATH:-"feature_repo"}
export AWS_XRAY_TRACING_DISABLED=${AWS_XRAY_TRACING_DISABLED:-"true"}
export SENTRY_DSN=${SENTRY_DSN:-""}
export SENTRY_TRACES_SAMPLE_RATE=${SENTRY_TRACES_SAMPLE_RATE:-"0.1"}
export ENVIRONMENT=${ENVIRONMENT:-"development"}

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️ Model file not found at $MODEL_PATH"
    echo "Creating a dummy model for testing..."

    mkdir -p models
    python -c "
import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Create a simple dummy model
model = RandomForestRegressor(n_estimators=10, random_state=42)
X_dummy = np.random.rand(100, 10)
y_dummy = np.random.rand(100)
model.fit(X_dummy, y_dummy)

# Save the model
with open('$MODEL_PATH', 'wb') as f:
    pickle.dump(model, f)

print('✅ Dummy model created')
"
fi

# Check if feature repo exists
if [ ! -d "$FEAST_REPO_PATH" ]; then
    echo "⚠️ Feature repo not found at $FEAST_REPO_PATH"
    echo "Run: ./scripts/init_feast.sh to initialize feature store"
fi

# Start X-Ray daemon if not disabled
if [ "$AWS_XRAY_TRACING_DISABLED" != "true" ]; then
    echo "🔍 Starting X-Ray daemon..."
    ./scripts/start_xray_daemon.sh 2000 &
    sleep 2
fi

# Run the API based on mode
case $MODE in
    "dev")
        echo "🔧 Running in development mode with auto-reload..."
        uvicorn src.app.main:app \
            --host 0.0.0.0 \
            --port $PORT \
            --reload \
            --log-level info
        ;;
    "prod")
        echo "🏭 Running in production mode..."
        uvicorn src.app.main:app \
            --host 0.0.0.0 \
            --port $PORT \
            --workers $WORKERS \
            --log-level warning \
            --access-log
        ;;
    "test")
        echo "🧪 Running in test mode..."
        uvicorn src.app.main:app \
            --host 127.0.0.1 \
            --port $PORT \
            --log-level debug &

        API_PID=$!
        sleep 5

        # Test the API
        echo "Testing API endpoints..."

        # Health check
        curl -f http://localhost:$PORT/health || echo "❌ Health check failed"

        # Root endpoint
        curl -f http://localhost:$PORT/ || echo "❌ Root endpoint failed"

        # Test prediction (will likely fail without proper model/features)
        curl -X POST http://localhost:$PORT/predict \
            -H "Content-Type: application/json" \
            -d '{"ticker":"AAPL"}' || echo "⚠️ Prediction test failed (expected)"

        # Test Sentry integration if DSN is set
        if [ -n "$SENTRY_DSN" ]; then
            echo "Testing Sentry integration..."
            curl -X POST http://localhost:$PORT/predict \
                -H "Content-Type: application/json" \
                -d '{"ticker":"INVALID_TICKER_FOR_SENTRY_TEST"}' || echo "✅ Sentry error test completed"
        fi

        # Stop the API
        kill $API_PID
        echo "✅ Test completed"
        ;;
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Usage: $0 [dev|prod|test] [port] [workers]"
        exit 1
        ;;
esac
