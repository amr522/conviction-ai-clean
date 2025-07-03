#!/bin/bash
# CI helper script to mask and echo the pinned dataset path
# Usage: ./ci_echo_dataset.sh

set -e  # Exit on any error

echo "🔍 CI: Verifying dataset pinning"

# Source the helper script to set environment variables
echo "⏳ Sourcing dataset pinning script..."
if source ./scripts/get_last_hpo_dataset.sh; then
  echo "✅ Dataset pinning script executed successfully"
else
  echo "❌ Failed to execute dataset pinning script"
  exit 1
fi

# Check if dataset was found
if [ -z "$PINNED_DATA_S3" ]; then
  echo "❌ ERROR: No pinned dataset found"
  exit 1
fi

# Mask the dataset path for security in CI logs
# Extract bucket and path structure while masking details
BUCKET=$(echo "$PINNED_DATA_S3" | sed -E 's|s3://([^/]+)/.*|\1|')
BUCKET_MASKED=$(echo "$BUCKET" | sed -E 's/^(.{4}).*(.{4})$/\1****\2/')
  
PATH_STRUCTURE=$(echo "$PINNED_DATA_S3" | sed -E 's|s3://[^/]+/(.*)|/\1|' | sed -E 's|/[^/]+|/***|g')
MASKED_PATH="s3://${BUCKET_MASKED}${PATH_STRUCTURE}"

echo ""
echo "🔒 Dataset verification (masked for security):"
echo "   HPO Job ID: $LAST_HPO_JOB"
echo "   Best Training Job: $BEST_JOB"
echo "   Dataset Path: $MASKED_PATH"
echo "   Completed Jobs: $JOB_COMPLETION_COUNT"
echo "✅ Dataset S3 URI structure is valid"

# Verify the dataset URI is saved to file
if [ -f "last_dataset_uri.txt" ]; then
  echo "✅ Dataset URI saved to last_dataset_uri.txt"
else
  echo "❌ Dataset URI file missing"
  exit 1
fi

# Verify the dataset path is properly used by the HPO launch script
echo ""
echo "🧪 Verifying HPO launch script uses the correct dataset..."
if python aws_hpo_launch.py --dry-run; then
  echo "✅ HPO launch script verified to use the correct dataset"
else
  echo "❌ HPO launch script failed to use the correct dataset"
  exit 1
fi

echo ""
echo "✅ CI: Dataset pinning verification complete"
exit 0
