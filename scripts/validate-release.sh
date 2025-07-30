#!/usr/bin/env bash
set -euo pipefail

# Validate release preparation
echo "🔍 Validating release setup..."

# Check required files
REQUIRED_FILES=(
    ".chglog/config.yml"
    ".chglog/CHANGELOG.tpl.md"
    "scripts/release.sh"
    "scripts/prepare-release.sh"
    ".github/workflows/release.yml"
    "CHANGELOG.md"
    "VERSION"
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
if [[ -x "scripts/release.sh" ]]; then
    echo "✅ release.sh is executable"
else
    echo "❌ release.sh not executable"
    exit 1
fi

if [[ -x "scripts/prepare-release.sh" ]]; then
    echo "✅ prepare-release.sh is executable"
else
    echo "❌ prepare-release.sh not executable"
    exit 1
fi

# Check git status
if [[ -n $(git status --porcelain) ]]; then
    echo "⚠️  Working directory has uncommitted changes"
else
    echo "✅ Working directory is clean"
fi

# Check current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "📍 Current branch: $CURRENT_BRANCH"

# Check if we have any tags
if git describe --tags --abbrev=0 &>/dev/null; then
    LAST_TAG=$(git describe --tags --abbrev=0)
    echo "🏷️  Last tag: $LAST_TAG"
else
    echo "🏷️  No tags found (first release)"
fi

# Check VERSION file content
if [[ -f VERSION ]]; then
    VERSION_CONTENT=$(cat VERSION)
    echo "📋 VERSION file content: $VERSION_CONTENT"
else
    echo "📋 VERSION file not found"
fi

echo ""
echo "🎉 Release validation completed!"
echo ""
echo "To create a release:"
echo "  ./scripts/release.sh                    # Auto-detect version"
echo "  ./scripts/release.sh v1.2.3            # Specific version"
echo "  ./scripts/prepare-release.sh v1.2.3    # Create release branch"
