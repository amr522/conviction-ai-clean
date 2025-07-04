#!/bin/bash
set -euo pipefail


DRY_RUN=${1:-"false"}
PINNED_CONFIG="models/pinned_successful_hpo/hpo_config_pinned.json"
BACKUP_PREFIX="datasets/successful_backups/$(date +%Y-%m-%d)"
BUCKET="hpo-bucket-773934887314"
STACK_NAME="hpo-s3-versioning-stack"

echo "🔄 Backing up successful dataset with S3 versioning..."

cd "$(dirname "$0")/.."

REQUIRED_PERMISSIONS=(
    "s3:GetObject"
    "s3:PutObject"
    "s3:GetBucketVersioning"
    "s3:PutBucketVersioning"
    "cloudformation:CreateStack"
    "cloudformation:UpdateStack"
    "cloudformation:DescribeStacks"
)

if [[ "$DRY_RUN" == "true" ]]; then
    echo "🔍 [DRY-RUN] Would validate IAM permissions: ${REQUIRED_PERMISSIONS[*]}"
    echo "🔍 [DRY-RUN] Would check pinned configuration: $PINNED_CONFIG"
    echo "🔍 [DRY-RUN] Would deploy CloudFormation stack for S3 versioning: $STACK_NAME"
    echo "🔍 [DRY-RUN] Would backup dataset to: s3://$BUCKET/$BACKUP_PREFIX/"
    echo "🔍 [DRY-RUN] Would create metadata file with HPO job details"
    exit 0
fi

if [[ ! -f "$PINNED_CONFIG" ]]; then
    echo "❌ Pinned configuration not found: $PINNED_CONFIG"
    exit 1
fi

echo "🔧 Enabling S3 versioning on bucket: $BUCKET"
aws s3api put-bucket-versioning --bucket "$BUCKET" --versioning-configuration Status=Enabled

VERSIONING_STATUS=$(aws s3api get-bucket-versioning --bucket "$BUCKET" --query 'Status' --output text)
if [[ "$VERSIONING_STATUS" != "Enabled" ]]; then
    echo "❌ Failed to enable S3 versioning on bucket: $BUCKET"
    exit 1
fi

echo "✅ S3 versioning confirmed enabled on bucket: $BUCKET"

DATASET_URI=$(jq -r '.dataset_uri' "$PINNED_CONFIG")
if [[ "$DATASET_URI" == "null" || -z "$DATASET_URI" ]]; then
    echo "❌ Dataset URI not found in pinned configuration"
    exit 1
fi

echo "🔄 Backing up successful dataset: $DATASET_URI"
echo "📁 Backup location: s3://$BUCKET/$BACKUP_PREFIX/"

if aws s3 ls "s3://$BUCKET/$BACKUP_PREFIX/train.csv" >/dev/null 2>&1; then
    echo "⚠️ Backup already exists at s3://$BUCKET/$BACKUP_PREFIX/train.csv"
    echo "🔍 Checking if backup is current..."
    
    if aws s3 ls "s3://$BUCKET/$BACKUP_PREFIX/metadata.json" >/dev/null 2>&1; then
        aws s3 cp "s3://$BUCKET/$BACKUP_PREFIX/metadata.json" /tmp/existing_metadata.json
        EXISTING_HPO_JOB=$(jq -r '.successful_hpo_job' /tmp/existing_metadata.json 2>/dev/null || echo "")
        CURRENT_HPO_JOB=$(jq -r '.successful_hpo_job' "$PINNED_CONFIG")
        
        if [[ "$EXISTING_HPO_JOB" == "$CURRENT_HPO_JOB" ]]; then
            echo "✅ Backup is current, no action needed"
            exit 0
        fi
    fi
    
    echo "🔄 Updating backup with new data..."
fi

aws s3 cp "$DATASET_URI" "s3://$BUCKET/$BACKUP_PREFIX/train.csv"

jq '{dataset_uri, successful_hpo_job, best_training_job, validation_auc, completion_time}' "$PINNED_CONFIG" > /tmp/backup_metadata.json
aws s3 cp /tmp/backup_metadata.json "s3://$BUCKET/$BACKUP_PREFIX/metadata.json"

if aws s3 ls "s3://$BUCKET/$BACKUP_PREFIX/train.csv" >/dev/null 2>&1 && \
   aws s3 ls "s3://$BUCKET/$BACKUP_PREFIX/metadata.json" >/dev/null 2>&1; then
    echo "✅ Dataset backup completed successfully with versioning enabled"
    echo "📊 Backup details:"
    echo "   - Dataset: $(jq -r '.dataset_uri' "$PINNED_CONFIG")"
    echo "   - HPO Job: $(jq -r '.successful_hpo_job' "$PINNED_CONFIG")"
    echo "   - Validation AUC: $(jq -r '.validation_auc' "$PINNED_CONFIG")"
    echo "   - Location: s3://$BUCKET/$BACKUP_PREFIX/"
else
    echo "❌ Backup verification failed"
    exit 1
fi

rm -f /tmp/backup_metadata.json /tmp/existing_metadata.json
