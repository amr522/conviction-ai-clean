#!/usr/bin/env bash
set -euo pipefail

# Test Docker image locally
IMAGE_NAME=${1:-"conviction-ai-pipeline"}
TAG=${2:-"latest"}

echo "🧪 Testing Docker image: $IMAGE_NAME:$TAG"

# Build image locally
echo "📦 Building image..."
docker build -t "$IMAGE_NAME:$TAG" .

# Test basic functionality
echo "🔍 Testing basic functionality..."
docker run --rm "$IMAGE_NAME:$TAG" --help

# Test with sample date (dry run)
echo "🏃 Testing pipeline execution (dry run)..."
docker run --rm \
    -e AWS_ACCESS_KEY_ID=test \
    -e AWS_SECRET_ACCESS_KEY=test \
    -e AWS_REGION=us-east-1 \
    "$IMAGE_NAME:$TAG" \
    --date 2025-01-01 \
    --dry-run

# Show image info
echo "📋 Image information:"
docker images "$IMAGE_NAME:$TAG" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

echo "✅ Docker image test completed successfully!"
echo "🚀 Run with: docker run --rm $IMAGE_NAME:$TAG --date YYYY-MM-DD"
