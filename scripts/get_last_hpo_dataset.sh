#!/bin/bash
# Script to get the S3 data source from the HPO job with exactly 138 completed training jobs
# Usage: source ./scripts/get_last_hpo_dataset.sh

# Find the HPO job with exactly 138 completed training jobs, sorted by most recent first
echo "Searching for HPO job with exactly 138 completed training jobs..."
aws sagemaker list-hyper-parameter-tuning-jobs \
  --max-items 200 \
  --query "HyperParameterTuningJobSummaries[?TrainingJobStatusCounters.Completed==\`138\`].[HyperParameterTuningJobName,CreationTime]" \
  --output text | sort -rk2 | head -n1 | awk '{print $1}' > LAST_HPO_JOB.txt
LAST_HPO_JOB=$(cat LAST_HPO_JOB.txt)
export LAST_HPO_JOB
echo "HPO job with 138 completed training jobs: $LAST_HPO_JOB"

# Get best training job
BEST_JOB=$(aws sagemaker describe-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name "$LAST_HPO_JOB" --query "BestTrainingJob.TrainingJobName" --output text)
export BEST_JOB
echo "Best training job: $BEST_JOB"

# Extract S3 data source URI
LAST_DATA_S3=$(aws sagemaker describe-training-job --training-job-name "$BEST_JOB" --query "InputDataConfig[0].DataSource.S3DataSource.S3Uri" --output text)
export LAST_DATA_S3
echo "Data source S3 URI: $LAST_DATA_S3"

# Export for both bash and zsh compatibility
if [ -n "$ZSH_VERSION" ]; then
  echo "export LAST_HPO_JOB=\"$LAST_HPO_JOB\"" >> ~/.zshrc
  echo "export BEST_JOB=\"$BEST_JOB\"" >> ~/.zshrc
  echo "export LAST_DATA_S3=\"$LAST_DATA_S3\"" >> ~/.zshrc
  echo "Environment variables also saved to ~/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
  echo "export LAST_HPO_JOB=\"$LAST_HPO_JOB\"" >> ~/.bashrc
  echo "export BEST_JOB=\"$BEST_JOB\"" >> ~/.bashrc
  echo "export LAST_DATA_S3=\"$LAST_DATA_S3\"" >> ~/.bashrc
  echo "Environment variables also saved to ~/.bashrc"
fi
