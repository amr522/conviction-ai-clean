#!/bin/bash
# run_stacking_pipeline.sh - Helper script to run the stacking pipeline

# Check if .env file exists, if not create it
if [ ! -f .env ]; then
  echo "Creating .env file..."
  echo "AWS_REGION=us-east-1" > .env
  echo "S3_BUCKET_NAME=your-bucket-name" >> .env
  echo "SAGEMAKER_EXECUTION_ROLE=your-role-arn" >> .env
  
  echo "Please edit the .env file with your AWS credentials and settings"
  exit 1
fi

# Source .env file
source .env

# Check required environment variables
if [ -z "$AWS_REGION" ] || [ -z "$S3_BUCKET_NAME" ] || [ -z "$SAGEMAKER_EXECUTION_ROLE" ]; then
  echo "Error: Missing required environment variables in .env file"
  echo "Please make sure AWS_REGION, S3_BUCKET_NAME, and SAGEMAKER_EXECUTION_ROLE are set"
  exit 1
fi

# Check if autopilot job name and deep model OOF prefix are provided as arguments
if [ $# -lt 2 ]; then
  echo "Usage: $0 <autopilot-job-name> <deep-model-oof-prefix>"
  echo "Example: $0 my-autopilot-job s3://my-bucket/deep-model-oof/"
  exit 1
fi

AUTOPILOT_JOB_NAME="$1"
DEEP_MODEL_OOF_PREFIX="$2"

echo "Running stacking pipeline..."
echo "  Autopilot Job: $AUTOPILOT_JOB_NAME"
echo "  Deep Model OOF Prefix: $DEEP_MODEL_OOF_PREFIX"
echo "  Region: $AWS_REGION"
echo "  S3 Bucket: $S3_BUCKET_NAME"
echo "  IAM Role: $SAGEMAKER_EXECUTION_ROLE"

# Run the stacking pipeline
./aws_pipeline/stacking_pipeline.sh \
  --region "$AWS_REGION" \
  --s3-bucket "$S3_BUCKET_NAME" \
  --autopilot-job-name "$AUTOPILOT_JOB_NAME" \
  --deep-model-oof-prefix "$DEEP_MODEL_OOF_PREFIX" \
  --iam-role-arn "$SAGEMAKER_EXECUTION_ROLE"

# Check if the stacking pipeline completed successfully
if [ $? -ne 0 ]; then
  echo "Error: Stacking pipeline failed"
  exit 1
fi

echo "Stacking pipeline completed successfully!"
echo "Endpoint information is in stacked_endpoint_info.json"
