#!/usr/bin/env bash
set -euo pipefail

# SBOM generation script for local development
IMAGE_NAME=${1:-"conviction-ai-pipeline:latest"}
OUTPUT_DIR=${2:-"./sbom"}

echo "📋 Generating SBOM for $IMAGE_NAME"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Install Syft if not present
if ! command -v syft &> /dev/null; then
    echo "Installing Syft..."
    curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
fi

# Install Grype if not present
if ! command -v grype &> /dev/null; then
    echo "Installing Grype..."
    curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
fi

echo "Tool versions:"
echo "Syft: $(syft --version)"
echo "Grype: $(grype --version)"

# Generate SBOM in multiple formats
echo "Generating CycloneDX SBOM..."
syft "$IMAGE_NAME" -o cyclonedx-json > "$OUTPUT_DIR/cyclonedx.json"

echo "Generating SPDX SBOM..."
syft "$IMAGE_NAME" -o spdx-json > "$OUTPUT_DIR/spdx.json"

echo "Generating human-readable SBOM..."
syft "$IMAGE_NAME" -o table > "$OUTPUT_DIR/sbom.txt"

# Validate SBOM with Grype
echo "Scanning SBOM for vulnerabilities..."
if grype "$OUTPUT_DIR/spdx.json" --fail-on high,critical; then
    echo "✅ SBOM validation passed - no HIGH/CRITICAL vulnerabilities"
else
    echo "❌ SBOM validation failed - HIGH/CRITICAL vulnerabilities detected"
    echo "📄 Detailed report:"
    grype "$OUTPUT_DIR/spdx.json" -o table
    exit 1
fi

echo "📋 SBOM generation completed:"
echo "  CycloneDX: $OUTPUT_DIR/cyclonedx.json"
echo "  SPDX: $OUTPUT_DIR/spdx.json"
echo "  Human-readable: $OUTPUT_DIR/sbom.txt"
