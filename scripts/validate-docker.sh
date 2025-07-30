#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Validating Docker setup..."

# Check required files
REQUIRED_FILES=(
    "Dockerfile"
    ".dockerignore"
    "scripts/publish_docker.sh"
    "scripts/test_docker.sh"
    "docker-compose.prod.yml"
    ".env.docker.example"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

# Check script permissions
if [[ -x "scripts/publish_docker.sh" ]]; then
    echo "✅ publish_docker.sh is executable"
else
    echo "❌ publish_docker.sh not executable"
    exit 1
fi

if [[ -x "scripts/test_docker.sh" ]]; then
    echo "✅ test_docker.sh is executable"
else
    echo "❌ test_docker.sh not executable"
    exit 1
fi

# Check Docker availability
if command -v docker &> /dev/null; then
    echo "✅ Docker is available"
    echo "  Version: $(docker --version)"
else
    echo "❌ Docker not found"
    exit 1
fi

# Check if Docker daemon is running
if docker info &> /dev/null; then
    echo "✅ Docker daemon is running"
else
    echo "❌ Docker daemon not running"
    exit 1
fi

# Check Dockerfile syntax (basic validation)
echo "🔍 Validating Dockerfile syntax..."
if docker build --no-cache -t validation-test . > /dev/null 2>&1; then
    echo "✅ Dockerfile builds successfully"
    docker rmi validation-test > /dev/null 2>&1 || true
else
    echo "❌ Dockerfile build failed"
    echo "Run 'docker build .' to see detailed errors"
    exit 1
fi

# Check environment variables
echo ""
echo "📋 Environment Configuration:"
echo "  DOCKER_REGISTRY: ${DOCKER_REGISTRY:-not set (will use docker.io)}"
echo "  DOCKER_USERNAME: ${DOCKER_USERNAME:-not set}"
echo "  DOCKER_PASSWORD: ${DOCKER_PASSWORD:-not set}"

if [[ -n "${DOCKER_USERNAME:-}" && -n "${DOCKER_PASSWORD:-}" ]]; then
    echo "✅ Docker credentials configured"
else
    echo "⚠️  Docker credentials not configured (required for publishing)"
fi

echo ""
echo "🎉 Docker validation completed!"
echo ""
echo "Next steps:"
echo "  ./scripts/test_docker.sh                    # Test build locally"
echo "  ./scripts/publish_docker.sh                # Publish to registry"
echo "  docker-compose -f docker-compose.prod.yml up  # Run with compose"
