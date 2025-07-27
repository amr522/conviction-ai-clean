#!/usr/bin/env bash
set -euo pipefail

# Start AWS X-Ray daemon for local development
DAEMON_PORT=${1:-2000}
DAEMON_CONFIG=${2:-""}

echo "🔍 Starting AWS X-Ray daemon"
echo "Port: $DAEMON_PORT"

# Check if daemon is already running
if pgrep -f "xray" > /dev/null; then
    echo "⚠️ X-Ray daemon is already running"
    echo "Process: $(pgrep -f xray)"
    exit 0
fi

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Download X-Ray daemon if not present
XRAY_DAEMON="xray_daemon"
if [[ "$OSTYPE" == "darwin"* ]]; then
    XRAY_DAEMON="xray_daemon_darwin"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    XRAY_DAEMON="xray_daemon_linux"
fi

if [ ! -f "$XRAY_DAEMON" ]; then
    echo "📥 Downloading X-Ray daemon..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        curl -o xray_daemon_darwin.zip https://s3.us-east-2.amazonaws.com/aws-xray-assets.us-east-2/xray-daemon/aws-xray-daemon-macos-3.x.zip
        unzip -o xray_daemon_darwin.zip
        chmod +x xray
        mv xray xray_daemon_darwin
        rm xray_daemon_darwin.zip
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -o xray_daemon_linux.zip https://s3.us-east-2.amazonaws.com/aws-xray-assets.us-east-2/xray-daemon/aws-xray-daemon-linux-3.x.zip
        unzip -o xray_daemon_linux.zip
        chmod +x xray
        mv xray xray_daemon_linux
        rm xray_daemon_linux.zip
    fi
fi

# Create daemon configuration if provided
if [ -n "$DAEMON_CONFIG" ]; then
    echo "📝 Using custom daemon configuration: $DAEMON_CONFIG"
    CONFIG_FLAG="-c $DAEMON_CONFIG"
else
    CONFIG_FLAG=""
fi

# Start X-Ray daemon
echo "🚀 Starting X-Ray daemon on port $DAEMON_PORT..."

./$XRAY_DAEMON \
    -o \
    -n us-east-1 \
    -b 127.0.0.1:$DAEMON_PORT \
    $CONFIG_FLAG \
    > logs/xray_daemon.log 2>&1 &

DAEMON_PID=$!
echo "✅ X-Ray daemon started with PID: $DAEMON_PID"

# Wait a moment and check if daemon is running
sleep 2
if kill -0 $DAEMON_PID 2>/dev/null; then
    echo "✅ X-Ray daemon is running successfully"
    echo "📊 Traces will be available at: https://console.aws.amazon.com/xray/home"
    echo "📝 Logs: logs/xray_daemon.log"
    echo ""
    echo "To stop the daemon:"
    echo "  kill $DAEMON_PID"
    echo ""
    echo "To test tracing:"
    echo "  python src/train_and_evaluate.py --start-date 2025-01-16 --end-date 2025-01-16 --dry-run"
else
    echo "❌ X-Ray daemon failed to start"
    echo "Check logs: logs/xray_daemon.log"
    exit 1
fi