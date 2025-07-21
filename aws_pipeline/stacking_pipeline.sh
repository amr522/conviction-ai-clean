#!/bin/bash
# stacking_pipeline.sh - Script for training a stacking model with SageMaker
# This script downloads out-of-fold predictions from AutoPilot and deep learning models,
# combines them into a stacking dataset, and trains a LightGBM model on top.

set -e  # Exit on error

# Default values
REGION=""
S3_BUCKET=""
AUTOPILOT_JOB_NAME=""
DEEP_MODEL_OOF_PREFIX=""
IAM_ROLE_ARN=""
JOB_NAME_PREFIX="StackModel"
INSTANCE_TYPE="ml.m5.xlarge"
INSTANCE_COUNT=1
VOLUME_SIZE=50

# Function to print usage information
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --region REGION                       AWS region"
  echo "  --s3-bucket BUCKET                    S3 bucket name"
  echo "  --autopilot-job-name JOB_NAME         SageMaker Autopilot job name"
  echo "  --deep-model-oof-prefix S3_PREFIX     S3 prefix for deep model OOF predictions"
  echo "  --iam-role-arn IAM_ROLE               SageMaker execution role ARN"
  echo "  --job-name-prefix PREFIX              Training job name prefix (default: StackModel)"
  echo "  --instance-type INSTANCE_TYPE         SageMaker instance type (default: ml.m5.xlarge)"
  echo "  --instance-count COUNT                Number of instances (default: 1)"
  echo "  --volume-size SIZE                    EBS volume size in GB (default: 50)"
  echo "  --help                                Show this help message"
  exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --region)
      REGION="$2"
      shift 2
      ;;
    --s3-bucket)
      S3_BUCKET="$2"
      shift 2
      ;;
    --autopilot-job-name)
      AUTOPILOT_JOB_NAME="$2"
      shift 2
      ;;
    --deep-model-oof-prefix)
      DEEP_MODEL_OOF_PREFIX="$2"
      shift 2
      ;;
    --iam-role-arn)
      IAM_ROLE_ARN="$2"
      shift 2
      ;;
    --job-name-prefix)
      JOB_NAME_PREFIX="$2"
      shift 2
      ;;
    --instance-type)
      INSTANCE_TYPE="$2"
      shift 2
      ;;
    --instance-count)
      INSTANCE_COUNT="$2"
      shift 2
      ;;
    --volume-size)
      VOLUME_SIZE="$2"
      shift 2
      ;;
    --help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Load environment variables if not specified via command line
if [ -f .env ]; then
  echo "Loading environment variables from .env file"
  source .env
fi

# Check for required parameters
if [ -z "$REGION" ]; then
  if [ -n "$AWS_REGION" ]; then
    REGION="$AWS_REGION"
  else
    echo "Error: --region or AWS_REGION environment variable is required"
    usage
  fi
fi

if [ -z "$S3_BUCKET" ]; then
  if [ -n "$S3_BUCKET_NAME" ]; then
    S3_BUCKET="$S3_BUCKET_NAME"
  else
    echo "Error: --s3-bucket or S3_BUCKET_NAME environment variable is required"
    usage
  fi
fi

if [ -z "$IAM_ROLE_ARN" ]; then
  if [ -n "$SAGEMAKER_EXECUTION_ROLE" ]; then
    IAM_ROLE_ARN="$SAGEMAKER_EXECUTION_ROLE"
  else
    echo "Error: --iam-role-arn or SAGEMAKER_EXECUTION_ROLE environment variable is required"
    usage
  fi
fi

if [ -z "$AUTOPILOT_JOB_NAME" ]; then
  echo "Error: --autopilot-job-name is required"
  usage
fi

if [ -z "$DEEP_MODEL_OOF_PREFIX" ]; then
  echo "Error: --deep-model-oof-prefix is required"
  usage
fi

echo "=============================================="
echo "Stacking Pipeline Configuration:"
echo "  AWS Region: $REGION"
echo "  S3 Bucket: $S3_BUCKET"
echo "  Autopilot Job: $AUTOPILOT_JOB_NAME"
echo "  Deep Model OOF Prefix: $DEEP_MODEL_OOF_PREFIX"
echo "  Instance Type: $INSTANCE_TYPE"
echo "  Instance Count: $INSTANCE_COUNT"
echo "=============================================="

# Create directories for data
mkdir -p autopilot-oof
mkdir -p deep-model-oof
mkdir -p stacking-data

# Step 1: Download Autopilot OOF predictions
echo "Downloading Autopilot OOF predictions..."
aws s3 cp "s3://$S3_BUCKET/automl-out/$AUTOPILOT_JOB_NAME/candidate-predictions/" ./autopilot-oof/ --recursive --region $REGION
if [ $? -ne 0 ]; then
  echo "Error: Failed to download Autopilot OOF predictions"
  exit 1
fi

# Step 2: Download deep model OOF predictions
echo "Downloading deep model OOF predictions..."
aws s3 cp "$DEEP_MODEL_OOF_PREFIX" ./deep-model-oof/ --recursive --region $REGION
if [ $? -ne 0 ]; then
  echo "Error: Failed to download deep model OOF predictions"
  exit 1
fi

# Step 3: Create stacking dataset by combining OOF predictions
echo "Creating stacking dataset..."
python -c "
import pandas as pd
import os
import glob
import numpy as np

# Find all prediction CSV files
autopilot_files = glob.glob('./autopilot-oof/**/predictions.csv', recursive=True)
deep_model_files = glob.glob('./deep-model-oof/**/predictions.csv', recursive=True)

if not autopilot_files:
    raise Exception('No AutoPilot prediction files found')
if not deep_model_files:
    raise Exception('No deep model prediction files found')

# Process each AutoPilot model's predictions
ap_dfs = []
for i, file_path in enumerate(autopilot_files):
    model_name = os.path.basename(os.path.dirname(file_path))
    df = pd.read_csv(file_path)
    
    # Rename prediction columns to indicate source model
    pred_cols = [col for col in df.columns if col.startswith('pred_')]
    rename_dict = {col: f'ap_{model_name}_{col}' for col in pred_cols}
    df = df.rename(columns=rename_dict)
    
    # Keep only id, target, and prediction columns
    keep_cols = ['id', 'target'] + list(rename_dict.values())
    df = df[keep_cols]
    
    ap_dfs.append(df)

# Process each deep model's predictions
deep_dfs = []
for i, file_path in enumerate(deep_model_files):
    model_name = os.path.basename(os.path.dirname(file_path))
    df = pd.read_csv(file_path)
    
    # Identify prediction columns (not id or target)
    pred_cols = [col for col in df.columns if col not in ['id', 'target']]
    rename_dict = {col: f'deep_{model_name}_{col}' for col in pred_cols}
    df = df.rename(columns=rename_dict)
    
    # Keep only id, target, and prediction columns
    keep_cols = ['id', 'target'] + list(rename_dict.values())
    df = df[keep_cols]
    
    deep_dfs.append(df)

# Start with the first AutoPilot model's predictions
result_df = ap_dfs[0]

# Merge with other AutoPilot models
for df in ap_dfs[1:]:
    result_df = result_df.merge(df, on=['id', 'target'], how='inner')

# Merge with deep models
for df in deep_dfs:
    result_df = result_df.merge(df, on=['id', 'target'], how='inner')

print(f'Merged dataset shape: {result_df.shape}')
print(f'Columns: {', '.join(result_df.columns)}')

# Write stacking dataset
result_df.to_csv('./stacking-data/stacking-data.csv', index=False)
print(f'Wrote {len(result_df)} rows to stacking-data.csv')
"

if [ $? -ne 0 ]; then
  echo "Error: Failed to create stacking dataset"
  exit 1
fi

# Step 4: Upload stacking dataset to S3
echo "Uploading stacking dataset to S3..."
aws s3 cp ./stacking-data/stacking-data.csv "s3://$S3_BUCKET/stacking-data/" --region $REGION
if [ $? -ne 0 ]; then
  echo "Error: Failed to upload stacking dataset to S3"
  exit 1
fi

# Step 5: Start SageMaker training job
echo "Starting SageMaker training job..."

# Generate timestamp for job name
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
JOB_NAME="${JOB_NAME_PREFIX}-${TIMESTAMP}"
echo "Job name: $JOB_NAME"

# Create and start training job
aws sagemaker create-training-job \
  --training-job-name $JOB_NAME \
  --algorithm-specification TrainingImage=438346466558.dkr.ecr.$REGION.amazonaws.com/lightgbm:1.3-1,TrainingInputMode=File \
  --role-arn $IAM_ROLE_ARN \
  --input-data-config "[{\"ChannelName\":\"train\",\"DataSource\":{\"S3DataSource\":{\"S3DataType\":\"S3Prefix\",\"S3Uri\":\"s3://$S3_BUCKET/stacking-data/\",\"S3DataDistributionType\":\"FullyReplicated\"}},\"ContentType\":\"text/csv\"}]" \
  --output-data-config S3OutputPath="s3://$S3_BUCKET/stacked-model/$JOB_NAME/output" \
  --resource-config InstanceType=$INSTANCE_TYPE,InstanceCount=$INSTANCE_COUNT,VolumeSizeInGB=$VOLUME_SIZE \
  --hyper-parameters "{\"objective\":\"regression\",\"metric\":\"rmse\",\"num_leaves\":\"64\",\"learning_rate\":\"0.05\"}" \
  --stopping-condition MaxRuntimeInSeconds=14400 \
  --region $REGION

if [ $? -ne 0 ]; then
  echo "Error: Failed to start SageMaker training job"
  exit 1
fi

echo "Training job $JOB_NAME started successfully"
echo "Waiting for training job to complete..."

# Step 6: Wait for training job to complete
aws sagemaker wait training-job-completed-or-stopped \
  --training-job-name $JOB_NAME \
  --region $REGION

if [ $? -ne 0 ]; then
  echo "Error: Failed while waiting for training job"
  exit 1
fi

# Check if job was successful
TRAINING_STATUS=$(aws sagemaker describe-training-job \
  --training-job-name $JOB_NAME \
  --query 'TrainingJobStatus' \
  --output text \
  --region $REGION)

if [ "$TRAINING_STATUS" != "Completed" ]; then
  echo "Error: Training job failed with status $TRAINING_STATUS"
  exit 1
fi

echo "Training job completed successfully"

# Step 7: Create model
echo "Creating SageMaker model..."
MODEL_NAME="${JOB_NAME}-model"
MODEL_DATA="s3://$S3_BUCKET/stacked-model/$JOB_NAME/output/model.tar.gz"

aws sagemaker create-model \
  --model-name $MODEL_NAME \
  --primary-container Image=438346466558.dkr.ecr.$REGION.amazonaws.com/lightgbm:1.3-1,ModelDataUrl=$MODEL_DATA \
  --execution-role-arn $IAM_ROLE_ARN \
  --region $REGION

if [ $? -ne 0 ]; then
  echo "Error: Failed to create SageMaker model"
  exit 1
fi

# Step 8: Create endpoint configuration
echo "Creating endpoint configuration..."
ENDPOINT_CONFIG_NAME="${JOB_NAME}-config"

aws sagemaker create-endpoint-config \
  --endpoint-config-name $ENDPOINT_CONFIG_NAME \
  --production-variants "[{\"VariantName\":\"AllTraffic\",\"ModelName\":\"$MODEL_NAME\",\"InitialInstanceCount\":1,\"InstanceType\":\"ml.m4.xlarge\"}]" \
  --region $REGION

if [ $? -ne 0 ]; then
  echo "Error: Failed to create endpoint configuration"
  exit 1
fi

# Step 9: Create endpoint
echo "Creating SageMaker endpoint..."
ENDPOINT_NAME="stacked-model-endpoint-${TIMESTAMP}"

aws sagemaker create-endpoint \
  --endpoint-name $ENDPOINT_NAME \
  --endpoint-config-name $ENDPOINT_CONFIG_NAME \
  --region $REGION

if [ $? -ne 0 ]; then
  echo "Error: Failed to create endpoint"
  exit 1
fi

echo "Waiting for endpoint to become available..."
aws sagemaker wait endpoint-in-service \
  --endpoint-name $ENDPOINT_NAME \
  --region $REGION

if [ $? -ne 0 ]; then
  echo "Error: Failed while waiting for endpoint"
  exit 1
fi

echo "Endpoint $ENDPOINT_NAME is now available"

# Save endpoint info to file
echo "Saving endpoint information..."
echo "{\"endpoint_name\": \"$ENDPOINT_NAME\", \"model_name\": \"$MODEL_NAME\"}" > stacked_endpoint_info.json

echo "=============================================="
echo "Stacking Pipeline Complete!"
echo "  Model: $MODEL_NAME"
echo "  Endpoint: $ENDPOINT_NAME"
echo "  Endpoint info saved to: stacked_endpoint_info.json"
echo "=============================================="
