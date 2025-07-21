#!/bin/bash
# create_stacked_endpoint_info.sh - Creates a mock stacked endpoint info file without training

# Exit on error
set -e

# Function to print usage
print_usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --autopilot-job-name NAME   SageMaker Autopilot job name"
  echo "  --s3-bucket BUCKET          S3 bucket name"
  echo "  --region REGION             AWS region (default: from environment)"
  echo "  --help                      Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --autopilot-job-name)
      AUTOPILOT_JOB_NAME="$2"
      shift 2
      ;;
    --s3-bucket)
      S3_BUCKET="$2"
      shift 2
      ;;
    --region)
      AWS_REGION="$2"
      shift 2
      ;;
    --help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

# Load environment variables if available
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
fi

# Check for required parameters
if [ -z "$AUTOPILOT_JOB_NAME" ]; then
    # Try to get from endpoint_info.json
    if [ -f "endpoint_info.json" ]; then
        AUTOPILOT_JOB_NAME=$(grep -o '"job_name": "[^"]*' endpoint_info.json | cut -d'"' -f4)
        echo "Using Autopilot job name from endpoint_info.json: $AUTOPILOT_JOB_NAME"
    else
        echo "Error: Autopilot job name not provided. Use --autopilot-job-name"
        print_usage
        exit 1
    fi
fi

# Set S3 bucket from environment if not provided
if [ -z "$S3_BUCKET" ]; then
    S3_BUCKET=$S3_BUCKET_NAME
    if [ -z "$S3_BUCKET" ]; then
        echo "Error: S3 bucket not provided. Use --s3-bucket"
        print_usage
        exit 1
    fi
fi

# Set AWS region from environment if not provided
if [ -z "$AWS_REGION" ]; then
    if [ -z "$AWS_REGION" ]; then
        echo "Error: AWS region not provided and not set in environment. Use --region"
        print_usage
        exit 1
    fi
fi

echo "Creating mock stacked endpoint info file..."
echo "Autopilot job name: $AUTOPILOT_JOB_NAME"
echo "S3 bucket: $S3_BUCKET"
echo "AWS region: $AWS_REGION"

# Create a timestamp for the mock endpoint
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ENDPOINT_NAME="stacked-model-endpoint-$TIMESTAMP"
MODEL_NAME="stacked-model-$TIMESTAMP"
TRAINING_JOB_NAME="StackModel-$TIMESTAMP"
MODEL_DATA_URL="s3://${S3_BUCKET}/stacked-model/${TRAINING_JOB_NAME}/output/model.tar.gz"
STACKING_DATA_S3_PREFIX="s3://${S3_BUCKET}/conviction-ai/stacking-data/"

# Save endpoint info to JSON file
cat > stacked_endpoint_info.json <<EOF
{
  "endpoint_name": "${ENDPOINT_NAME}",
  "model_name": "${MODEL_NAME}",
  "training_job_name": "${TRAINING_JOB_NAME}",
  "model_data_url": "${MODEL_DATA_URL}",
  "stacking_data_s3_prefix": "${STACKING_DATA_S3_PREFIX}",
  "status": "MOCK_ENDPOINT_CREATED",
  "autopilot_job_name": "${AUTOPILOT_JOB_NAME}",
  "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

echo "Mock stacked endpoint information saved to stacked_endpoint_info.json"
echo "This file simulates the information that would be created by a successful stacking pipeline."
echo "In a production environment, a real endpoint would be created using SageMaker."
