#!/usr/bin/env bash
set -euo pipefail

# Helper script to prepare a release branch
VERSION=${1:-}

if [[ -z "$VERSION" ]]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 v1.2.3"
    exit 1
fi

# Validate version format
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format vX.Y.Z (e.g., v1.2.3)"
    exit 1
fi

BRANCH_NAME="release/${VERSION}"

echo "🚀 Preparing release branch: $BRANCH_NAME"

# Check if we're on main
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "main" ]]; then
    echo "Error: Must be on main branch. Current: $CURRENT_BRANCH"
    exit 1
fi

# Check for clean working directory
if [[ -n $(git status --porcelain) ]]; then
    echo "Error: Working directory not clean. Commit changes first."
    exit 1
fi

# Pull latest changes
git pull origin main

# Create release branch
git checkout -b "$BRANCH_NAME"

# Update version in files
echo "${VERSION#v}" > VERSION
echo "✅ Updated VERSION file"

# Update README badge
if [[ -f README.md ]]; then
    sed -i.bak "s/version-[0-9]\+\.[0-9]\+\.[0-9]\+-blue/version-${VERSION#v}-blue/g" README.md
    rm -f README.md.bak
    echo "✅ Updated README version badge"
fi

# Commit version updates
git add VERSION README.md
git commit -m "chore: prepare release $VERSION"

# Push release branch
git push origin "$BRANCH_NAME"

echo "✅ Release branch $BRANCH_NAME created and pushed"
echo "🔗 Create PR: https://github.com/your-organization/conviction-ai-clean/compare/main...$BRANCH_NAME"
echo ""
echo "Next steps:"
echo "1. Create PR from $BRANCH_NAME to main"
echo "2. Review and merge PR"
echo "3. Release workflow will automatically trigger"
