#!/bin/bash

set -euo pipefail

TARGET_COMPLETED=${TARGET_COMPLETED:-138}
REGION=${AWS_DEFAULT_REGION:-us-east-1}
OUTPUT_FILE="last_dataset_uri.txt"

echo "🔍 Searching for HPO jobs with ≥${TARGET_COMPLETED} completed training jobs..."

JOBS=$(aws sagemaker list-hyper-parameter-tuning-jobs \
    --region "$REGION" \
    --creation-time-after "$(date -d '30 days ago' -u +%Y-%m-%dT%H:%M:%SZ)" \
    --sort-by CreationTime \
    --sort-order Descending \
    --query 'HyperParameterTuningJobSummaries[?HyperParameterTuningJobStatus==`Completed`].[HyperParameterTuningJobName,TrainingJobStatusCounters.Completed]' \
    --output text)

if [[ -z "$JOBS" ]]; then
    echo "❌ No completed HPO jobs found in the last 30 days"
    exit 1
fi

LAST_HPO_JOB=""
LAST_DATA_S3=""

while IFS=$'\t' read -r job_name completed_count; do
    if [[ -n "$completed_count" && "$completed_count" -ge "$TARGET_COMPLETED" ]]; then
        echo "✅ Found qualifying job: $job_name (${completed_count} completed jobs)"
        
        JOB_DETAILS=$(aws sagemaker describe-hyper-parameter-tuning-job \
            --region "$REGION" \
            --hyper-parameter-tuning-job-name "$job_name" \
            --query 'TrainingJobDefinition.InputDataConfig[0].DataSource.S3DataSource.S3Uri' \
            --output text)
        
        if [[ "$JOB_DETAILS" != "None" && -n "$JOB_DETAILS" ]]; then
            LAST_HPO_JOB="$job_name"
            LAST_DATA_S3="$JOB_DETAILS"
            break
        else
            echo "⚠️ Job $job_name has no valid S3 data source, skipping..."
        fi
    else
        echo "⏭️ Skipping $job_name (only ${completed_count:-0} completed jobs, need ≥${TARGET_COMPLETED})"
    fi
done <<< "$JOBS"

if [[ -z "$LAST_HPO_JOB" ]]; then
    echo "❌ No HPO job found with ≥${TARGET_COMPLETED} completed training jobs"
    echo "Available jobs and their completion counts:"
    echo "$JOBS"
    exit 1
fi

if [[ -z "$LAST_DATA_S3" ]]; then
    echo "❌ No valid S3 data source found for job: $LAST_HPO_JOB"
    exit 1
fi

if [[ ! "$LAST_DATA_S3" =~ ^s3://[a-zA-Z0-9.-]+/.+ ]]; then
    echo "❌ Invalid S3 URI format: $LAST_DATA_S3"
    exit 1
fi

echo "🎯 PINNED DATASET SELECTION:"
echo "HPO Job: $LAST_HPO_JOB"
echo "Data Source: $LAST_DATA_S3"

echo "$LAST_DATA_S3" > "$OUTPUT_FILE"
export PINNED_DATA_S3="$LAST_DATA_S3"

echo "💾 Dataset URI saved to: $OUTPUT_FILE"
echo "🔒 PINNED_DATA_S3 exported: $PINNED_DATA_S3"

echo "🔍 Verifying S3 access..."
if aws s3 ls "$LAST_DATA_S3" >/dev/null 2>&1; then
    echo "✅ S3 dataset accessible"
else
    echo "⚠️ Warning: S3 dataset may not be accessible with current credentials"
fi

echo "🚀 HPO dataset pinning complete!"
