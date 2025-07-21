#!/bin/bash
# run_aws_pipeline.sh - Run the entire AWS pipeline from data preparation to model analysis

# Exit on error
set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment variables
if [ -f ../.env ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' ../.env | xargs)
elif [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found. Run setup_aws_env.sh first."
    exit 1
fi

# Check for required environment variables
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ] || [ -z "$AWS_REGION" ]; then
    echo "Error: AWS credentials not set. Check your .env file."
    exit 1
fi

if [ -z "$SAGEMAKER_EXECUTION_ROLE" ]; then
    echo "Error: SAGEMAKER_EXECUTION_ROLE not set. Check your .env file."
    exit 1
fi

# Check if SageMaker role has the required S3 permissions
echo "Checking SageMaker execution role permissions..."
ROLE_NAME=$(echo $SAGEMAKER_EXECUTION_ROLE | sed 's/.*role\///')
POLICY_CHECK=$(aws iam list-role-policies --role-name $ROLE_NAME 2>/dev/null | grep -E "S3AccessForSageMaker|SageMakerS3Access" || echo "")

if [ -z "$POLICY_CHECK" ]; then
    echo "Warning: SageMaker execution role may not have sufficient S3 permissions."
    echo "If the job fails with S3 access errors, ensure the role has s3:PutObject, s3:GetObject, and s3:ListBucket permissions."
    echo "See sagemaker-s3-policy.json for the required permissions."
    echo "Continue anyway? (y/n)"
    read -r CONTINUE
    if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Default values
S3_BUCKET=${S3_BUCKET_NAME:-"sagemaker-us-east-1-773934887314"}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-2}
MAX_CANDIDATES=${MAX_CANDIDATES:-10}
INSTANCE_TYPE=${INSTANCE_TYPE:-"ml.m5.large"}
TARGET_COLUMN=${TARGET_COLUMN:-"return"}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --s3-bucket)
      S3_BUCKET="$2"
      shift 2
      ;;
    --max-runtime)
      MAX_RUNTIME_HOURS="$2"
      shift 2
      ;;
    --max-candidates)
      MAX_CANDIDATES="$2"
      shift 2
      ;;
    --instance-type)
      INSTANCE_TYPE="$2"
      shift 2
      ;;
    --target-column)
      TARGET_COLUMN="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --s3-bucket BUCKET    S3 bucket name (default: $S3_BUCKET)"
      echo "  --max-runtime HOURS   Maximum runtime in hours (default: $MAX_RUNTIME_HOURS)"
      echo "  --max-candidates N    Maximum number of model candidates (default: $MAX_CANDIDATES)"
      echo "  --instance-type TYPE  Instance type for deployment (default: $INSTANCE_TYPE)"
      echo "  --target-column COL   Target column to predict (default: $TARGET_COLUMN)"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Step 1: Convert Parquet to CSV if needed
echo "Step 1: Checking if data conversion is needed..."
TRAIN_DATA_URI="s3://${S3_BUCKET}/conviction-ai/clean/train_dataset/"
CSV_DATA_URI="s3://${S3_BUCKET}/conviction-ai/csv/train_dataset/"

# You can uncomment this if you need to convert Parquet to CSV
# echo "Converting Parquet to CSV..."
# python "$SCRIPT_DIR/convert_parquet_to_csv.py" \
#   --source-uri "$TRAIN_DATA_URI" \
#   --dest-uri "$CSV_DATA_URI" \
#   --region "$AWS_REGION"

# Step 2: Run SageMaker Autopilot
echo "Step 2: Training model with SageMaker Autopilot..."
python "$SCRIPT_DIR/run_sagemaker_autopilot.py" \
  --role-arn "$SAGEMAKER_EXECUTION_ROLE" \
  --bucket "$S3_BUCKET" \
  --input-s3-uri "$TRAIN_DATA_URI" \
  --output-s3-uri "s3://${S3_BUCKET}/conviction-ai/automl-out/" \
  --max-candidates "$MAX_CANDIDATES" \
  --max-runtime-hours "$MAX_RUNTIME_HOURS" \
  --instance-type "$INSTANCE_TYPE" \
  --target-column "$TARGET_COLUMN" \
  --region "$AWS_REGION"

# Get the endpoint name from the generated JSON file
if [ -f ../endpoint_info.json ]; then
  ENDPOINT_NAME=$(grep -o '"endpoint_name": "[^"]*' ../endpoint_info.json | cut -d'"' -f4)
  AUTOPILOT_JOB_NAME=$(grep -o '"job_name": "[^"]*' ../endpoint_info.json | cut -d'"' -f4)
  echo "Trained model deployed to endpoint: $ENDPOINT_NAME"
  echo "Autopilot job name: $AUTOPILOT_JOB_NAME"
elif [ -f endpoint_info.json ]; then
  ENDPOINT_NAME=$(grep -o '"endpoint_name": "[^"]*' endpoint_info.json | cut -d'"' -f4)
  AUTOPILOT_JOB_NAME=$(grep -o '"job_name": "[^"]*' endpoint_info.json | cut -d'"' -f4)
  echo "Trained model deployed to endpoint: $ENDPOINT_NAME"
  echo "Autopilot job name: $AUTOPILOT_JOB_NAME"
else
  echo "Error: endpoint_info.json not found. SageMaker AutoML job may have failed."
  exit 1
fi

# Step 3: Run the stacking pipeline
echo "Step 3: Running stacking pipeline to combine AutoML and deep learning models..."
DEEP_MODEL_OOF_S3_PREFIX="s3://${S3_BUCKET}/conviction-ai/deep-model-oof/"
STACKING_DATA_S3_PREFIX="s3://${S3_BUCKET}/conviction-ai/stacking-data/"

# Make script executable if not already
chmod +x "$SCRIPT_DIR/stacking_pipeline.sh"

"$SCRIPT_DIR/stacking_pipeline.sh" \
  --autopilot-job-name "$AUTOPILOT_JOB_NAME" \
  --deep-model-oof-prefix "$DEEP_MODEL_OOF_S3_PREFIX" \
  --stacking-data-prefix "$STACKING_DATA_S3_PREFIX" \
  --iam-role-arn "$SAGEMAKER_EXECUTION_ROLE" \
  --s3-bucket "$S3_BUCKET" \
  --region "$AWS_REGION"

# Get stacked model endpoint name
if [ -f ../stacked_endpoint_info.json ]; then
  STACKED_ENDPOINT_NAME=$(grep -o '"endpoint_name": "[^"]*' ../stacked_endpoint_info.json | cut -d'"' -f4)
  echo "Stacked model deployed to endpoint: $STACKED_ENDPOINT_NAME"
elif [ -f stacked_endpoint_info.json ]; then
  STACKED_ENDPOINT_NAME=$(grep -o '"endpoint_name": "[^"]*' stacked_endpoint_info.json | cut -d'"' -f4)
  echo "Stacked model deployed to endpoint: $STACKED_ENDPOINT_NAME"
else
  echo "Warning: stacked_endpoint_info.json not found. Stacking pipeline may have failed."
  # Continue with the pipeline even if stacking failed
fi

# Step 4: Analyze the models for data leakage and overfitting
echo "Step 4: Analyzing models for data leakage and overfitting..."
if [ -f "$SCRIPT_DIR/run_model_analysis.sh" ]; then
  chmod +x "$SCRIPT_DIR/run_model_analysis.sh"
  
  "$SCRIPT_DIR/run_model_analysis.sh" \
    --endpoint-name "$ENDPOINT_NAME" \
    --train-s3 "$TRAIN_DATA_URI" \
    --output-dir "../model_analysis_results/autopilot" \
    --target-column "$TARGET_COLUMN" \
    --region "$AWS_REGION"

  # Also analyze the stacked model if available
  if [ ! -z "$STACKED_ENDPOINT_NAME" ]; then
    "$SCRIPT_DIR/run_model_analysis.sh" \
      --endpoint-name "$STACKED_ENDPOINT_NAME" \
      --train-s3 "$TRAIN_DATA_URI" \
      --output-dir "../model_analysis_results/stacked" \
      --target-column "$TARGET_COLUMN" \
      --region "$AWS_REGION"
  fi
else
  echo "Warning: run_model_analysis.sh not found. Skipping model analysis."
fi

echo "Pipeline completed successfully!"
echo "Analysis results are available in the model_analysis_results directory"
