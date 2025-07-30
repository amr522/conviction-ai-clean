#!/usr/bin/env bash
set -euo pipefail

# Demo script to show release process (dry run)
echo "🎬 Demo: Automated Release Process"
echo "=================================="

# Show current state
echo ""
echo "📋 Current State:"
echo "  Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "  Version: $(cat VERSION 2>/dev/null || echo 'Not set')"
echo "  Last tag: $(git describe --tags --abbrev=0 2>/dev/null || echo 'None')"

# Show what would happen in release
echo ""
echo "🔍 Release Analysis:"

# Check for commits since last tag
OLD_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
COMMITS_SINCE_TAG=$(git rev-list ${OLD_TAG}..HEAD --count 2>/dev/null || echo "0")

if [[ $COMMITS_SINCE_TAG -eq 0 ]]; then
    echo "  ⚠️  No commits since last tag - nothing to release"
else
    echo "  📊 Commits since $OLD_TAG: $COMMITS_SINCE_TAG"

    # Analyze commit types
    echo "  📝 Recent commits:"
    git log ${OLD_TAG}..HEAD --oneline --max-count=5 | sed 's/^/    /'

    # Determine version bump
    if git log ${OLD_TAG}..HEAD --oneline | grep -q "^feat\|BREAKING CHANGE"; then
        BUMP_TYPE="minor (feature added)"
        NEXT_TAG=$(echo $OLD_TAG | awk -F. '{$2++; $3=0; print $1"."$2"."$3}' | sed 's/^v/v/')
    elif git log ${OLD_TAG}..HEAD --oneline | grep -q "^fix\|^perf"; then
        BUMP_TYPE="patch (fix/performance)"
        NEXT_TAG=$(echo $OLD_TAG | awk -F. '{$3++; print $1"."$2"."$3}' | sed 's/^v/v/')
    else
        BUMP_TYPE="patch (default)"
        NEXT_TAG=$(echo $OLD_TAG | awk -F. '{$3++; print $1"."$2"."$3}' | sed 's/^v/v/')
    fi

    echo "  🎯 Suggested version: $OLD_TAG → $NEXT_TAG ($BUMP_TYPE)"
fi

echo ""
echo "🚀 Release Commands:"
echo "  ./scripts/release.sh                    # Auto-detect version"
echo "  ./scripts/release.sh v1.2.3            # Specific version"
echo "  ./scripts/prepare-release.sh v1.2.3    # Create release branch"

echo ""
echo "📚 Commit Message Examples:"
echo "  feat: add new performance optimization  # Minor version bump"
echo "  fix: resolve memory leak in pipeline   # Patch version bump"
echo "  perf: optimize join operations         # Patch version bump"
echo "  refactor: consolidate utilities        # Patch version bump"

echo ""
echo "✅ Demo completed! Use the commands above to create an actual release."
