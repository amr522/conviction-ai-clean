#!/usr/bin/env bash
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
IMAGE_NAME=${1:-"${DOCKER_REGISTRY:-docker.io/myuser}/conviction-ai-pipeline"}
TAG=${2:-$(git describe --tags --abbrev=0 2>/dev/null || echo "latest")}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"docker.io"}

log_info "Building Docker image $IMAGE_NAME:$TAG"

# Validate Git tag format if not latest
if [[ "$TAG" != "latest" && ! "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    log_warn "Tag '$TAG' doesn't follow semantic versioning (vX.Y.Z)"
fi

# Build image with build args
log_info "Building image with tag: $TAG"
docker build \
    --build-arg VERSION="$TAG" \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VCS_REF="$(git rev-parse HEAD)" \
    -t "$IMAGE_NAME:$TAG" \
    .

# Tag as latest if this is a version tag
if [[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    log_info "Tagging as latest..."
    docker tag "$IMAGE_NAME:$TAG" "$IMAGE_NAME:latest"
    PUSH_LATEST=true
else
    PUSH_LATEST=false
fi

# Login to registry if credentials provided
if [[ -n "${DOCKER_USERNAME:-}" && -n "${DOCKER_PASSWORD:-}" ]]; then
    log_info "Logging into registry $DOCKER_REGISTRY..."
    echo "${DOCKER_PASSWORD}" | docker login "$DOCKER_REGISTRY" --username "$DOCKER_USERNAME" --password-stdin
else
    log_warn "No Docker credentials provided. Assuming already logged in."
fi

# Push images
log_info "Pushing $IMAGE_NAME:$TAG..."
docker push "$IMAGE_NAME:$TAG"

if [[ "$PUSH_LATEST" == "true" ]]; then
    log_info "Pushing $IMAGE_NAME:latest..."
    docker push "$IMAGE_NAME:latest"
fi

log_info "✅ Published $IMAGE_NAME:$TAG"
if [[ "$PUSH_LATEST" == "true" ]]; then
    log_info "✅ Published $IMAGE_NAME:latest"
fi

# Output image info
log_info "📋 Image Information:"
echo "  Registry: $DOCKER_REGISTRY"
echo "  Image: $IMAGE_NAME"
echo "  Tag: $TAG"
echo "  Size: $(docker images --format 'table {{.Size}}' "$IMAGE_NAME:$TAG" | tail -n1)"
echo "  Pull command: docker pull $IMAGE_NAME:$TAG"