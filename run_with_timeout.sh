#!/bin/bash
# run_with_timeout.sh - Execute a command with a timeout
# Usage: ./run_with_timeout.sh <timeout_in_seconds> <command> [args...]

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <timeout_in_seconds> <command> [args...]"
  exit 1
fi

TIMEOUT=$1
shift
COMMAND="$@"
JOB_NAME=$(echo "$COMMAND" | awk '{print $1}' | xargs basename)

echo "⏱️ Running command with $TIMEOUT second timeout: $COMMAND"

# Function to monitor the process
monitor_process() {
  local pid=$1
  local timeout=$2
  local start_time=$SECONDS
  local elapsed=0
  
  while kill -0 $pid 2>/dev/null; do
    # Check if we've exceeded the timeout
    elapsed=$((SECONDS - start_time))
    if [ $elapsed -gt $timeout ]; then
      echo "❌ Command timed out after ${elapsed}s: $COMMAND"
      kill -9 $pid 2>/dev/null || true
      return 1
    fi
    
    # Show progress every minute
    if [ $((elapsed % 60)) -eq 0 ] && [ $elapsed -gt 0 ]; then
      echo "⏳ Still running after ${elapsed}s..."
    fi
    
    sleep 5
  done
  
  return 0
}

# Start the command in background
eval "$COMMAND" &
CMD_PID=$!

# Monitor the process
if monitor_process $CMD_PID $TIMEOUT; then
  # Wait for the process to complete
  wait $CMD_PID
  EXIT_CODE=$?
  echo "✅ Command completed after $((SECONDS))s with exit code $EXIT_CODE"
  exit $EXIT_CODE
else
  echo "📝 Log entry: JOB $JOB_NAME timed out after ${TIMEOUT}s"
  exit 124  # Standard timeout exit code
fi
