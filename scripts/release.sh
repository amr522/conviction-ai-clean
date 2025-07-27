#!/usr/bin/env bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if we're on main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "main" ]]; then
    log_error "Must be on main branch to release. Current branch: $CURRENT_BRANCH"
    exit 1
fi

# Check for uncommitted changes
if [[ -n $(git status --porcelain) ]]; then
    log_error "Working directory is not clean. Commit or stash changes first."
    exit 1
fi

# Get current version
OLD_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
log_info "Last tag: $OLD_TAG"

# Determine next version
if [[ $# -eq 0 ]]; then
    # Auto-determine version based on commit messages
    COMMITS_SINCE_TAG=$(git rev-list ${OLD_TAG}..HEAD --count)
    if [[ $COMMITS_SINCE_TAG -eq 0 ]]; then
        log_warn "No commits since last tag. Nothing to release."
        exit 0
    fi

    # Simple semantic versioning based on commit messages
    if git log ${OLD_TAG}..HEAD --oneline | grep -q "^feat\|BREAKING CHANGE"; then
        # Minor version bump for features
        NEXT_TAG=$(echo $OLD_TAG | awk -F. '{$2++; $3=0; print $1"."$2"."$3}' | sed 's/^v/v/')
    elif git log ${OLD_TAG}..HEAD --oneline | grep -q "^fix\|^perf"; then
        # Patch version bump for fixes
        NEXT_TAG=$(echo $OLD_TAG | awk -F. '{$3++; print $1"."$2"."$3}' | sed 's/^v/v/')
    else
        # Default patch bump
        NEXT_TAG=$(echo $OLD_TAG | awk -F. '{$3++; print $1"."$2"."$3}' | sed 's/^v/v/')
    fi
else
    NEXT_TAG=$1
fi

log_info "Next version: $NEXT_TAG"

# Check if git-chglog is available
if ! command -v git-chglog &> /dev/null; then
    log_warn "git-chglog not found. Installing via go..."
    if command -v go &> /dev/null; then
        go install github.com/git-chglog/git-chglog/cmd/git-chglog@latest
    else
        log_error "git-chglog not found and go not available. Please install git-chglog manually."
        exit 1
    fi
fi

# Generate changelog
log_info "Generating changelog..."
git-chglog --next-tag $NEXT_TAG -o CHANGELOG.md

# Update VERSION file
echo "${NEXT_TAG#v}" > VERSION
log_info "Updated VERSION file to ${NEXT_TAG#v}"

# Update version badge in README
if [[ -f README.md ]]; then
    sed -i.bak "s/version-[0-9]\+\.[0-9]\+\.[0-9]\+-blue/version-${NEXT_TAG#v}-blue/g" README.md
    rm -f README.md.bak
    log_info "Updated version badge in README.md"
fi

# Commit changelog and version updates
log_info "Committing changelog and version updates..."
git add CHANGELOG.md VERSION README.md
git commit -m "chore: release $NEXT_TAG

- Update CHANGELOG.md with latest changes
- Bump version to ${NEXT_TAG#v}
- Update README version badge"

# Create and push tag
log_info "Creating tag $NEXT_TAG..."
git tag -a $NEXT_TAG -m "Release $NEXT_TAG

$(git log ${OLD_TAG}..HEAD --oneline | head -5)"

# Push to origin
log_info "Pushing to origin..."
git push origin main --tags

log_info "✅ Release $NEXT_TAG completed successfully!"
log_info "🔗 View release: https://github.com/your-organization/conviction-ai-clean/releases/tag/$NEXT_TAG"
