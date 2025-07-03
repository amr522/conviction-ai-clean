#!/bin/bash
# Script to get the S3 data source from the HPO job with at least a target number of completed training jobs
# Usage: source ./scripts/get_last_hpo_dataset.sh
# Or with custom threshold: TARGET_COMPLETED=142 source ./scripts/get_last_hpo_dataset.sh

# Allow parameterization of the minimum completed jobs threshold
TARGET_COMPLETED=${TARGET_COMPLETED:-138}

# Create a temporary directory for output files
TEMP_DIR=$(mktemp -d)
JOB_LIST_FILE="${TEMP_DIR}/job_list.txt"

# Function for cleanup on exit
cleanup() {
  rm -rf "${TEMP_DIR}"
  echo "Temporary files cleaned up"
}

# Set trap for cleanup
trap cleanup EXIT

# Find the HPO job with at least TARGET_COMPLETED completed training jobs, sorted by most recent first
echo "Searching for HPO jobs with ≥${TARGET_COMPLETED} completed training jobs..."
aws sagemaker list-hyper-parameter-tuning-jobs \
  --max-items 200 \
  --query "HyperParameterTuningJobSummaries[?TrainingJobStatusCounters.Completed>=\`${TARGET_COMPLETED}\`].[HyperParameterTuningJobName,CreationTime,TrainingJobStatusCounters.Completed]" \
  --output text > "${JOB_LIST_FILE}"

# Count how many jobs were found
JOB_COUNT=$(wc -l < "${JOB_LIST_FILE}")
if [[ $JOB_COUNT -eq 0 ]]; then
  echo "❌ No HPO jobs found with ≥${TARGET_COMPLETED} completed training jobs"
  echo "Try lowering the TARGET_COMPLETED value"
  # Get count of jobs with any completed training jobs
  ALL_JOBS_COUNT=$(aws sagemaker list-hyper-parameter-tuning-jobs \
    --max-items 200 \
    --query "length(HyperParameterTuningJobSummaries)" \
    --output text)
  echo "There are ${ALL_JOBS_COUNT} HPO jobs in total"
  if [[ -n "$ZSH_VERSION" || -n "$BASH_VERSION" ]]; then
    return 1
  else
    exit 1
  fi
fi

echo "✅ Found ${JOB_COUNT} HPO jobs with ≥${TARGET_COMPLETED} completed training jobs"

# Get the most recent job
LAST_HPO_JOB=$(sort -rk2 "${JOB_LIST_FILE}" | head -n1 | awk '{print $1}')
JOB_COMPLETION_COUNT=$(grep "${LAST_HPO_JOB}" "${JOB_LIST_FILE}" | awk '{print $3}')

# Fail fast if no job found
if [[ -z "$LAST_HPO_JOB" ]]; then
  echo "❌ No HPO job found with ≥$TARGET_COMPLETED completed models"
  if [[ -n "$ZSH_VERSION" || -n "$BASH_VERSION" ]]; then
    return 1
  else
    exit 1
  fi
fi

export LAST_HPO_JOB
echo "✅ Selected most recent HPO job with ${JOB_COMPLETION_COUNT} completed training jobs: ${LAST_HPO_JOB}"

# Get best training job
BEST_JOB=$(aws sagemaker describe-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name "$LAST_HPO_JOB" --query "BestTrainingJob.TrainingJobName" --output text)

# Fail fast if no best job found
if [[ -z "$BEST_JOB" || "$BEST_JOB" == "None" ]]; then
  echo "❌ No best training job found for HPO job: $LAST_HPO_JOB"
  # Try to get any completed job from this HPO job as a fallback
  echo "Attempting to find any completed training job as fallback..."
  FALLBACK_JOB=$(aws sagemaker list-training-jobs-for-hyper-parameter-tuning-job \
    --hyper-parameter-tuning-job-name "$LAST_HPO_JOB" \
    --status-equals "Completed" \
    --sort-by "CreationTime" \
    --sort-order "Descending" \
    --max-results 1 \
    --query "TrainingJobSummaries[0].TrainingJobName" \
    --output text)
  
  if [[ -z "$FALLBACK_JOB" || "$FALLBACK_JOB" == "None" ]]; then
    echo "❌ No completed training jobs found for HPO job: $LAST_HPO_JOB"
    if [[ -n "$ZSH_VERSION" || -n "$BASH_VERSION" ]]; then
      return 1
    else
      exit 1
    fi
  else
    BEST_JOB="$FALLBACK_JOB"
    echo "⚠️ Using fallback completed training job: $BEST_JOB"
  fi
fi

export BEST_JOB
echo "✅ Best training job: $BEST_JOB"

# Extract S3 data source URI
LAST_DATA_S3=$(aws sagemaker describe-training-job --training-job-name "$BEST_JOB" --query "InputDataConfig[0].DataSource.S3DataSource.S3Uri" --output text)

# Fail fast if no data source URI found
if [[ -z "$LAST_DATA_S3" || "$LAST_DATA_S3" == "None" || ! "$LAST_DATA_S3" =~ ^s3:// ]]; then
  echo "❌ Invalid or missing S3 data source URI for training job: $BEST_JOB"
  if [[ -n "$ZSH_VERSION" || -n "$BASH_VERSION" ]]; then
    return 1
  else
    exit 1
  fi
fi

export LAST_DATA_S3
echo "✅ Data source S3 URI: $LAST_DATA_S3"

# Explicitly pin this dataset URI for future use
export PINNED_DATA_S3="$LAST_DATA_S3"
echo "🔒 Pinned dataset URI: $PINNED_DATA_S3"

# Save the pinned dataset URI to a file for persistence
echo "$PINNED_DATA_S3" > "$(dirname "$0")/../last_dataset_uri.txt"
echo "✅ Saved pinned dataset URI to last_dataset_uri.txt"

# Verify the dataset exists in S3
echo "🔍 Verifying dataset exists in S3..."
if aws s3 ls "$PINNED_DATA_S3" &>/dev/null; then
  echo "✅ Dataset verified in S3: $PINNED_DATA_S3"
else
  echo "⚠️ WARNING: Could not verify dataset at $PINNED_DATA_S3"
  echo "   This might be due to permissions or the dataset location has changed"
fi

# Export for both bash and zsh compatibility
if [ -n "$ZSH_VERSION" ]; then
  # Check if variables already exist in .zshrc
  if ! grep -q "export LAST_HPO_JOB=" ~/.zshrc; then
    echo "export LAST_HPO_JOB=\"$LAST_HPO_JOB\"" >> ~/.zshrc
    echo "export BEST_JOB=\"$BEST_JOB\"" >> ~/.zshrc
    echo "export LAST_DATA_S3=\"$LAST_DATA_S3\"" >> ~/.zshrc
    echo "export PINNED_DATA_S3=\"$PINNED_DATA_S3\"" >> ~/.zshrc
    echo "✅ Environment variables saved to ~/.zshrc"
  else
    echo "⚠️ Environment variables already exist in ~/.zshrc"
    echo "   To update, edit ~/.zshrc manually or remove the existing entries first"
  fi
elif [ -n "$BASH_VERSION" ]; then
  # Check if variables already exist in .bashrc
  if ! grep -q "export LAST_HPO_JOB=" ~/.bashrc; then
    echo "export LAST_HPO_JOB=\"$LAST_HPO_JOB\"" >> ~/.bashrc
    echo "export BEST_JOB=\"$BEST_JOB\"" >> ~/.bashrc
    echo "export LAST_DATA_S3=\"$LAST_DATA_S3\"" >> ~/.bashrc
    echo "export PINNED_DATA_S3=\"$PINNED_DATA_S3\"" >> ~/.bashrc
    echo "✅ Environment variables saved to ~/.bashrc"
  else
    echo "⚠️ Environment variables already exist in ~/.bashrc"
    echo "   To update, edit ~/.bashrc manually or remove the existing entries first"
  fi
fi

# Print summary
echo ""
echo "🎉 SUCCESS: Dataset path synchronized"
echo "HPO Job:         $LAST_HPO_JOB"
echo "Completed Jobs:  $JOB_COMPLETION_COUNT out of ${TARGET_COMPLETED}+ required"
echo "Best Job:        $BEST_JOB"
echo "Data Source:     $LAST_DATA_S3"
echo "Pinned Dataset:  $PINNED_DATA_S3"
echo ""
echo "To use in Python scripts:"
echo "import os"
echo "data_source = os.getenv('PINNED_DATA_S3') or os.getenv('LAST_DATA_S3')"
echo ""
echo "To use in shell scripts:"
echo "echo \"Using pinned dataset: \$PINNED_DATA_S3\""
echo ""
echo "To update the source in the future, run this command again with:"
echo "source ./scripts/get_last_hpo_dataset.sh"
echo "or specify a different minimum job count:"
echo "TARGET_COMPLETED=150 source ./scripts/get_last_hpo_dataset.sh"
