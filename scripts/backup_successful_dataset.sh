#!/bin/bash
set -euo pipefail

PINNED_CONFIG="models/pinned_successful_hpo/hpo_config_pinned.json"
BACKUP_PREFIX="datasets/successful_backups/$(date +%Y-%m-%d)"

if [[ ! -f "$PINNED_CONFIG" ]]; then
    echo "❌ Pinned configuration not found: $PINNED_CONFIG"
    exit 1
fi

DATASET_URI=$(jq -r '.dataset_uri' "$PINNED_CONFIG")
BUCKET="hpo-bucket-773934887314"

echo "🔄 Backing up successful dataset: $DATASET_URI"
echo "📁 Backup location: s3://$BUCKET/$BACKUP_PREFIX/"

aws s3 cp "$DATASET_URI" "s3://$BUCKET/$BACKUP_PREFIX/train.csv"

jq '{dataset_uri, successful_hpo_job, best_training_job, validation_auc, completion_time}' "$PINNED_CONFIG" > /tmp/backup_metadata.json
aws s3 cp /tmp/backup_metadata.json "s3://$BUCKET/$BACKUP_PREFIX/metadata.json"

echo "✅ Dataset backup completed with versioning enabled"
