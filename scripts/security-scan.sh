#!/usr/bin/env bash
set -euo pipefail

# Local security scanning script
IMAGE_NAME=${1:-"conviction-ai-pipeline:latest"}

echo "🔒 Running security scan on $IMAGE_NAME"

# Install security tools if not present
if ! command -v trivy &> /dev/null; then
    echo "Installing Trivy..."
    curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
fi

if ! command -v bandit &> /dev/null; then
    echo "Installing Python security tools..."
    pip install bandit safety
fi

echo "Security tools versions:"
echo "Trivy: $(trivy --version)"
echo "Bandit: $(bandit --version)"
echo "Safety: $(safety --version)"

# Scan for HIGH and CRITICAL vulnerabilities
echo "Scanning for HIGH and CRITICAL vulnerabilities..."
trivy image \
    --severity HIGH,CRITICAL \
    --exit-code 1 \
    --ignore-unfixed \
    "$IMAGE_NAME"

TRIVY_EXIT=$?

# Run Bandit scan
echo "Running Bandit static code analysis..."
bandit -r src/ -lll
BANDIT_EXIT=$?

# Run Safety scan
echo "Running Safety dependency scan..."
pip freeze > requirements.txt
safety check
SAFETY_EXIT=$?

# Source Slack notification helper
source "$(dirname "$0")/slack_notify.sh"

# Prepare status report
TRIVY_STATUS="PASS"; [ $TRIVY_EXIT -ne 0 ] && TRIVY_STATUS="FAIL"
BANDIT_STATUS="PASS"; [ $BANDIT_EXIT -ne 0 ] && BANDIT_STATUS="FAIL"
SAFETY_STATUS="PASS"; [ $SAFETY_EXIT -ne 0 ] && SAFETY_STATUS="FAIL"

REPORT="Trivy: $TRIVY_STATUS | Bandit: $BANDIT_STATUS | Safety: $SAFETY_STATUS"

# Check all results
if [ $TRIVY_EXIT -eq 0 ] && [ $BANDIT_EXIT -eq 0 ] && [ $SAFETY_EXIT -eq 0 ]; then
    echo "✅ All security scans passed"
    echo "  ✅ Trivy: No HIGH/CRITICAL container vulnerabilities"
    echo "  ✅ Bandit: No high-severity code issues"
    echo "  ✅ Safety: No dependency vulnerabilities"
    notify_security "PASSED" "$REPORT"
else
    echo "❌ Security scan failed:"
    [ $TRIVY_EXIT -ne 0 ] && echo "  ❌ Trivy: Container vulnerabilities detected"
    [ $BANDIT_EXIT -ne 0 ] && echo "  ❌ Bandit: Code security issues detected"
    [ $SAFETY_EXIT -ne 0 ] && echo "  ❌ Safety: Dependency vulnerabilities detected"
    notify_security "FAILED" "$REPORT"
    exit 1
fi