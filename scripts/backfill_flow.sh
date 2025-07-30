#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting distributed backfill with Dask cluster"

# Start Dask scheduler in background
echo "📡 Starting Dask scheduler..."
dask-scheduler --port 8786 --dashboard-address :8787 &
SCHEDULER_PID=$!

# Wait for scheduler to start
sleep 3

# Start Dask workers in background
echo "👷 Starting Dask workers..."
dask-worker tcp://127.0.0.1:8786 --nprocs 4 --nthreads 2 &
WORKER_PID=$!

# Wait for workers to connect
sleep 5

echo "✅ Dask cluster ready"
echo "📊 Dashboard available at: http://localhost:8787"

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up Dask cluster..."
    kill $WORKER_PID 2>/dev/null || true
    kill $SCHEDULER_PID 2>/dev/null || true
    wait $WORKER_PID 2>/dev/null || true
    wait $SCHEDULER_PID 2>/dev/null || true
    echo "✅ Cleanup completed"
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Run the backfill flow with distributed processing
echo "🔄 Running distributed backfill flow..."
python src/flows/historical_backfill_flow.py --distributed "$@"

echo "🎉 Distributed backfill completed successfully"